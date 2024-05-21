#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle

import torch
from torch import tensor
from sampling import *
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torcheval.metrics import FrechetInceptionDistance


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=True, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    else:
        data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def fed_avg(local_weights, dataset_size_per_client):
    avg_dict = {}
    sum_dataset = sum(dataset_size_per_client)
    for i, dictionary in enumerate(local_weights):
        for key, tensor in dictionary.items():
            if key not in avg_dict:
                avg_dict[key] = tensor.clone() * (dataset_size_per_client[i]/ sum_dataset)
            else:
                avg_dict[key] += tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
    return avg_dict


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def reg_loss_fn():
    mse = nn.MSELoss(reduction='sum')
    return lambda input, output: (
        mse(input, output)
    )


def kl_loss():
    return lambda z_dist: (
        kl_divergence(z_dist,
                      Normal(
                          torch.zeros_like(z_dist.mean),
                          torch.ones_like(z_dist.stddev)
                      )
        ).sum(-1).sum())


def vae_loss_fn(beta):
    """
    Loss function for the VAE to use for backpropagation. It considers two terms:
    - reconstruction loss by comparing image quality between input and output
    - difference between the current and desired latent probability distribution, computed with
    Kullback-Leibler divergence (KL)
    """
    reg = reg_loss_fn()
    kl_div = kl_loss()
    return lambda input, output, z_dist: \
        beta * reg(input, output) + \
        kl_div(z_dist)


def vae_classifier_loss_fn(alpha, beta):
    """
    Loss function for the VAE with classification to use for backpropagation. It considers three terms:
    - reconstruction loss by comparing image quality between input and output
    - difference between the current and desired latent probability distribution, computed with
    Kullback-Leibler divergence (KL)
    - Cross-entropy loss to minimize error between the actual and predicted outcomes
    """
    vl_fn = vae_loss_fn(beta)
    cl_fn = nn.CrossEntropyLoss()

    return lambda input, output, z_dist, labels: \
        vl_fn(input, output[0], z_dist) + \
        alpha * cl_fn(output[1], labels)


def frechet_inception_distance(real_x: tensor, syn_x: tensor) -> tensor:
    assert real_x.shape == syn_x.shape
    fid = FrechetInceptionDistance(feature_dim=2048)
    fid.update(real_x, is_real=True)
    fid.update(syn_x, is_real=False)
    return fid.compute()

