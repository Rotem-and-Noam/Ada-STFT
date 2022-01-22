#!/bin/bash

python train.py --test_name basic
python train.py --test_name learn_window --learn_window 1
python train.py --test_name learn_kernels --learn_kernels 1
python train.py --test_name learn_window_kernels --learn_kernels 1 --learn_window 1
python train.py --test_name learn_3_window_kernels --learn_kernels 1 --learn_window 1 --three_widows 1
python train.py --test_name learn_3_window --learn_window 1 --three_widows 1
python train.py --test_name learn_3_kernels --learn_kernels 1 --three_widows 1
