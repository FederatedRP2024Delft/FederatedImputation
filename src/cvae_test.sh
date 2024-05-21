#!/usr/bin/env bash

python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=10 --iid=1 --dirichlet=1.0 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
