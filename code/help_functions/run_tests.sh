#!/bin/bash

python ../train.py --test_name run_basic
python ../train.py --test_name run_aug --augmentation 1
python ../train.py --test_name run_learn_window --learn_window 1
python ../train.py --test_name run_learn_window_aug --learn_window 1 --augmentation 1
python ../train.py --test_name run_learn_kernels --learn_kernels 1
python ../train.py --test_name run_learn_kernels_aug --learn_kernels 1 --augmentation 1
python ../train.py --test_name run_learn_window_kernels --learn_kernels 1 --learn_window 1
python ../train.py --test_name run_learn_window_kernels_aug --learn_kernels 1 --learn_window 1 --augmentation 1
python ../train.py --test_name run_learn_3_window_kernels_aug --learn_kernels 1 --learn_window 1 --augmentation 1 --three_widows 1
python ../train.py --test_name run_learn_3_window_aug --learn_window 1 --augmentation 1 --three_widows 1
python ../train.py --test_name run_learn_3_kernels_aug --learn_kernels 1 --augmentation 1 --three_widows 1
python ../train.py --test_name run_basic_SGD --optimizer_class SGD
python ../train.py --test_name run_SGD_aug --optimizer_class SGD --augmentation 1
python ../train.py --test_name run_SGD_window_aug --optimizer_class SGD --augmentation 1 --learn_window 1
python ../train.py --test_name run_SGD_window_kernels_aug --optimizer_class SGD --augmentation 1 --learn_kernels 1 --learn_window 1
python ../train.py --test_name run_SGD_kernels_aug --optimizer_class SGD --augmentation 1 --learn_kernels 1
