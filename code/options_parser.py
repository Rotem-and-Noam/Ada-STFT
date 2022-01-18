import argparse
import json


def get_options():

    # loading training options and hyper-parameters
    with open("options.json", 'r') as fp:
        options = json.load(fp)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default=options['test_name'])
    parser.add_argument('--resume', type=int, default=options['resume'])
    parser.add_argument('--ckpt_test_name', type=str, default=options['ckpt_test_name'])
    parser.add_argument('--ckpt_interval', type=int, default=options['ckpt_interval'])
    parser.add_argument('--tensorboard_dir', type=str, default=options['tensorboard_dir'])
    parser.add_argument('--data_dir', type=str, default=options['data_dir'])
    parser.add_argument('--ckpt_dir', type=str, default=options['ckpt_dir'])
    parser.add_argument('--learn_window', type=int, default=options['learn_window'])
    parser.add_argument('--learn_kernels', type=int, default=options['learn_kernels'])
    parser.add_argument('--batch_size', type=int, default=options['batch_size'])
    parser.add_argument('--num_workers', type=int, default=options['num_workers'])
    parser.add_argument('--epoch_num', type=int, default=options['epoch_num'])
    parser.add_argument('--learning_rate', type=int, default=options['learning_rate'])
    parser.add_argument('--split_parts', type=int, default=options['split_parts'])
    parser.add_argument('--gamma', type=int, default=options['gamma'])
    parser.add_argument('--cpu', type=int, default=options['cpu'])
    parser.add_argument('--augmentation', type=int, default=options['augmentation'])
    parser.add_argument('--three_widows', type=int, default=options['three_widows'])
    parser.add_argument('--optimizer_class', type=str, default=options['optimizer_class'])

    return parser
