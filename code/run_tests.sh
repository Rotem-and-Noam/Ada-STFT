#!/bin/bash

python ./train_env.py --test_name run_basic
python ./train_env.py --test_name run_aug --augmentation 1
python ./train_env.py --test_name run_learn_window --learn_window 1
python ./train_env.py --test_name run_learn_window_aug --learn_window 1 --augmentation 1
python ./train_env.py --test_name run_learn_kernels --learn_kernels 1
python ./train_env.py --test_name run_learn_kernels_aug --learn_kernels 1 --augmentation 1
python ./train_env.py --test_name run_learn_window_kernels --learn_kernels 1 --learn_window 1
python ./train_env.py --test_name run_learn_window_kernels_aug --learn_kernels 1 --learn_window 1 --augmentation 1
