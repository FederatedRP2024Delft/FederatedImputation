#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import torch
import torch.nn.functional as F
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from impute import impute_naive, impute_cvae_naive
from utils import vae_classifier_loss_fn, vae_loss_fn
from vae.mnist_vae import VaeAutoencoderClassifier, ConditionalVae


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        if args.model == 'exq' or args.model=='resnet':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        random.shuffle(idxs)
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        train_dataset = DatasetSplit(dataset, idxs_train)
        # trained_vae = VaeAutoencoderClassifier(dim_encoding=2)
        # trained_vae.load_state_dict(torch.load("C:\\Users\\LohithSai\\Desktop\\FederatedImputation\\vae_data"
        #                                        f"\\models\\vae_{self.args.dirichlet}.pth"))
        trained_cvae = ConditionalVae(dim_encoding=3)

        checkpoint = torch.load(f'/home/neo/projects/FederatedImputation/vae_data/models/0_cvae_{self.args.dirichlet}.pth')
        trained_cvae.load_state_dict(checkpoint)

        generated_train_dataset = impute_cvae_naive(k=self.args.num_generate, trained_cvae=trained_cvae, initial_dataset=train_dataset)
        generated_train_dataset = [(torch.tensor(image), torch.tensor(label)) for image, label in
                                           generated_train_dataset]

        trainloader = DataLoader(generated_train_dataset,
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.model == 'resnet':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        if self.args.model == 'vae' or model is isinstance(model, VaeAutoencoderClassifier):
            # print(f"length train: {len(self.trainloader.dataset)}")
            local_losses = model.train_model(self.trainloader.dataset, epochs=self.args.local_ep)[1]
            # print(f"losses: {local_losses}")
            loss = np.mean(local_losses)
            # print(loss)
            return model.state_dict(), loss
        if self.args.model == 'cvae':
            # print(f"length train cvae: {len(self.trainloader.dataset)}")
            # print(type(model))
            local_losses = model.train_model(
    training_data=self.trainloader.dataset,
    batch_size=32,
    epochs=self.args.local_ep,
    learning_rate=0.001
)[1]
            loss = np.mean(local_losses)
            # print(loss)
            return model.state_dict(), loss
        else:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    if images.size(0) == 1:
                        # print("eval")
                        model.eval()
                    log_probs = model(images)
                    # Switch back to train mode
                    if images.size(0) == 1:
                        model.train()
                    # print(f"log probs? {log_probs}")
                    # print(f"sum: {sum(log_probs)}")

                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    # if self.args.verbose and (batch_idx % 10 == 0):
                    #     print(f"Loss: {loss.item()}")
                    #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         global_round, iter, batch_idx * len(images),
                    #         len(self.trainloader.dataset),
                    #         100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            if self.args.model == 'cvae':
                for i, image in enumerate(images):
                    vl_fn = vae_loss_fn(beta=1.0)
                    output = model(image, labels[i])
                    loss += vl_fn(image, output, model.z_dist)
                    total += 1
                continue

            else:
                outputs = model(images)
            if self.args.model == 'vae' or model is isinstance(model, VaeAutoencoderClassifier):
                complete_loss_fn = vae_classifier_loss_fn(model.alpha, model.beta)
                loss += complete_loss_fn(images, outputs, model.z_dist, labels)
                _, pred_labels = torch.max(outputs[1], 1)
                pred_labels = pred_labels.view(-1)
                # print(f"pred labels: {pred_labels}, labels: {labels}")
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            else:
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    pred_labels_list = []
    true_labels_list = []

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        if args.model == 'cvae':
            for i, image in enumerate(images):
                vl_fn = vae_loss_fn(beta=1.0)
                output = model(image, labels[i])
                loss += vl_fn(image, output, model.z_dist)
                total += 1
            continue
        else:
            outputs = model(images)

        if args.model == 'vae' or model is isinstance(model, VaeAutoencoderClassifier):
            complete_loss_fn = vae_classifier_loss_fn(model.alpha, model.beta)
            loss += complete_loss_fn(images, outputs, model.z_dist, labels)/len(testloader)
            _, pred_labels = torch.max(outputs[1], 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            pred_labels_list.append(pred_labels.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())
        else:
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()/len(testloader)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    # pred_labels_all = np.concatenate(pred_labels_list, axis=0)
    # true_labels_all = np.concatenate(true_labels_list, axis=0)
    # f1_macro = f1_score(true_labels_all, pred_labels_all, average='macro')
    # f1_micro = f1_score(true_labels_all, pred_labels_all, average='micro')
    accuracy = correct/total
    # return accuracy, loss, f1_macro, f1_micro

    return accuracy, loss