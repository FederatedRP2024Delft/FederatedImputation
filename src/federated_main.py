#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.4 --frac=1.0 --num_users=10 --local_ep=10



import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from models import ResNet, Bottleneck
from image_classifier.image_classifier import MNISTClassifier
from vae.mnist_vae import VaeAutoencoderClassifier, ConditionalVae
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ExquisiteNetV2, ExquisiteNetV1
from utils import get_dataset, average_weights, exp_details, fed_avg
import torchvision

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    elif args.model == 'vae':
        global_model = VaeAutoencoderClassifier(dim_encoding=2)

    elif args.model == 'cvae':
        global_model = ConditionalVae(dim_encoding=3)

    elif args.model == 'exq':
        global_model = ExquisiteNetV1(class_num=10, img_channels=1)

    elif args.model == 'resnet':
        global_model = ResNet(block=Bottleneck,
               layers=[2, 2, 2, 2],
               num_classes=10,
               grayscale=True)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    test_losses_per_client, test_accuracies_per_client = np.zeros((args.num_users, args.epochs)), np.zeros((args.num_users, args.epochs))
    train_losses_per_client = np.zeros((args.num_users, args.epochs))

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        dataset_size_per_client = [len(user_groups[i]) for i in idxs_users]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            model_copy = None
            if (args.model == 'vae' or args.model == 'cvae'):
                model_copy = type(global_model)()  # create a new instance of the same model
                model_copy.load_state_dict(global_model.state_dict())
            else:
                model_copy = copy.deepcopy(global_model)
            w, loss = local_model.update_weights(
                model=model_copy, global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            train_losses_per_client[idx][epoch] = loss
            # print(f"actual loss: {loss}")
            if np.isnan(loss):
                # print("loss was nan!!!!!!!!!!!!!!!")
                loss = local_losses[-1] if len(local_losses) > 0 else 0
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = fed_avg(local_weights, dataset_size_per_client)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        # print("sum local losses: ", sum(local_losses))
        # print("Num of local losses: ", len(local_losses))
        train_losses.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            test_losses_per_client[c][epoch] = loss
            test_accuracies_per_client[c][epoch] = acc
            # print("Accuracy: ", acc)
            # print("Loss: ", loss)
        train_accuracies.append(sum(list_acc) / len(list_acc))



        # print(f"IID data total communication rounds {i} accuracy: ", accuracy)

        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
            # print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # if cvae, test against classifier
        if args.model == 'cvae':
            # load data
            num_data = 60000
            training_data = torchvision.datasets.FashionMNIST(root='../../data/FMNIST_train', train=True, download=True,
                                                              transform=torchvision.transforms.ToTensor())
            testing_data = torchvision.datasets.FashionMNIST(root='../../data/FMNIST_test', train=False, download=True,
                                                             transform=torchvision.transforms.ToTensor())

            input = training_data.data[:num_data] / 255.0  # convert to float
            labels = training_data.targets[:num_data]

            # load classifier
            model = "classifier"
            dataset = "fmnist"
            batch_size = 64
            epoch = 10

            classifier = MNISTClassifier(input_size=784, num_classes=10)
            classifier.train_model(training_data, batch_size=batch_size, epochs=epoch)
            accuracy = classifier.test_model(testing_data)
            print("Test accuracy: ", accuracy)

            data_count = 10000
            ratios = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            images = []
            labels = []

            for label_idx, ratio in enumerate(ratios):
                num_samples_to_generate = int(data_count * ratio)
                images.append(
                    global_model.generate_data(n_samples=num_samples_to_generate, target_label=label_idx).cpu().detach())

                label = torch.zeros((num_samples_to_generate, 10), device=device)
                label[:, label_idx] = 1
                labels.append(label.cpu().detach())

            final_images = torch.vstack(images)
            final_labels = torch.vstack(labels)

            assert final_images.shape[0] == final_labels.shape[0]

            accuracy = classifier.test_model_syn_img_label(final_images, final_labels)
            print(f"IID-id {args.iid}, Dirichlet beta {args.dirichlet}, communication round {epoch + 1}, accuracy: ", accuracy)

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print("Train loss: ", train_losses)
        print("Train accuracy: ", train_accuracies)
        print("Test loss: ", test_losses)
        print("Test accuracy: ", test_accuracies)
    # print(f"trainloss: {train_loss}")
    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '/home/neo/projects/FederatedImputation/save/objects/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_losses_per_client, test_losses_per_client, test_accuracies_per_client], f)

    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    #
    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/neo/projects/FederatedImputation/save/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/neo/projects/FederatedImputation/save/cvae_{}_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.num_users, args.dirichlet, args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

    torch.save(global_model.state_dict(), f"/home/neo/projects/RP_data/models/federated_{args.model}_{args.dataset}_{args.iid}_{args.dirichlet}_{args.epochs}_{args.local_ep}_{args.num_users}.pt")
    # torch.save(global_model, f"/home/neo/projects/RP_data/models/cvae_{args.num_generate}_{args.model}_{args.dirichlet}_cvae.pt")