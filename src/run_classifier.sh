#!/usr/bin/env bash

python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=10 --dirichlet=0.1 --frac=1.0 --num_users=3 --local_ep=1 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0