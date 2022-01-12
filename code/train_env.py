import json
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from check_points import *
from get_components import *


class Env:

    def __init__(self, batch_size, num_workers, epoch_num, learning_rate, gamma, writer,
                 data_dir, ckpt, ckpt_interval, options, load_resnet_weight_path=None,
                 split_parts=1, learn_window=0, learn_kernels=0, **kwargs):
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']
        self.train_data = get_dataloader(mode='train', data_dir=data_dir, genres=self.genres,
                                         batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.val_data = get_dataloader(mode='val', data_dir=data_dir, genres=self.genres,
                                       batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.test_data = get_dataloader(mode='test', data_dir=data_dir, genres=self.genres,
                                        batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.batch_size = batch_size
        self.data_length = len(self.train_data)
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = get_model(self.device, ckpt, load_resnet_weight_path)
        self.optimizer, self.scheduler = get_optimizer(self.model, learning_rate, gamma, ckpt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.split_parts = split_parts
        self.epoch_num = epoch_num
        self.start_epoch = ckpt.start_epoch
        self.class_number = len(self.genres)
        self.options = options
        self.ckpt = ckpt
        self.ckpt_interval = ckpt_interval
        if learn_window:
            self.model.stft.learn_window()
        if learn_kernels:
            self.model.stft.learn_kernels()
        self.model.stft.print_learnable_params()
        print(f"Learnable parameters {self.count_parameters(self.model)}")
        print(f"Starting training on device {str(self.device)} for {str(self.epoch_num)} epochs")

    def train(self):
        for epoch in range(self.start_epoch, self.epoch_num):

            train_loss = self.train_epoch()
            val_accuracy, confusion_matrix, val_loss = self.calculate_accuracy_and_loss()

            print(f"epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
                  f"train loss: {train_loss:.5f}",
                  f"val loss: {val_loss:.5f}",
                  f"learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # send documentation to tensorboard
            self.tensorboard_logging(confusion_matrix, train_loss, val_loss, val_accuracy, epoch)
            # save check points
            if epoch % self.ckpt_interval == self.ckpt_interval - 1:
                self.ckpt.save_ckpt(self.model, self.optimizer, self.scheduler, epoch, self.options)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def tensorboard_logging(self, confusion_matrix, train_loss, val_loss, val_accuracy, epoch):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        self.writer.add_scalar('Window Change', self.model.stft.calc_window_change().item(), epoch)
        self.writer.add_scalar('Kernel Change', self.model.stft.calc_kernels_change().item(), epoch)
        if epoch % self.ckpt_interval == 0:
            self.writer.add_figure('confusion matrix', self.show_confusion_matrix(confusion_matrix), epoch)
            self.writer.add_figure('Window',
                                   self.show_window(self.model.stft.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                   epoch)

    def show_confusion_matrix(self, confusion_matrix, show=False):
        if show:
            plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0,
                   vmax=len(self.val_data)*self.batch_size/self.class_number, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(self.class_number), self.genres)
        plt.xlabel('Predicted Category')
        plt.xticks(range(self.class_number), self.genres)
        if show:
            plt.show()
        return fig

    def show_window(self, window):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.stem(window)
        ax.set_title("Window Coefficients")
        return fig

    def train_epoch(self):
        self.model.train()
        train_epoch_loss = 0
        samples_total = 0
        with tqdm(total=self.data_length) as pbar:
            for j, sample in enumerate(self.train_data):
                waveforms, labels = sample

                # getting batch output and calculating batch loss
                output = self.model(waveforms.to(self.device))
                loss = self.criterion(output, labels.to(self.device))

                # the three musketeers:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # updating parameters for calculating total loss
                train_epoch_loss += loss.detach().item() * labels.size(0)
                samples_total += labels.size(0)
                pbar.update()

        # calculating mean train loss for epoch
        train_loss = train_epoch_loss / samples_total
        self.scheduler.step()

        return train_loss

    def calculate_accuracy_and_loss(self, data_type='val'):
        self.model.eval()
        correct_total = 0
        samples_total = 0
        loss_total = 0
        confusion_matrix = np.zeros([self.class_number, self.class_number], int)

        if data_type == 'val':
            data = self.val_data
        elif data_type == 'test':
            data = self.test_data
        elif data_type == 'train':
            data = self.train_data
        else:
            raise Exception("data_type parameter is not one of following: val, test, train")

        # iterate on the test or validation set and calculate accuracy and loss per batch
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                labels_expanded = labels.to(self.device).repeat(self.split_parts, 1).T.reshape(-1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_expanded)
                loss_total += loss.item() * len(labels)
                _, predictions = torch.max(outputs.data, 1)
                predictions = predictions.reshape(-1, self.split_parts)
                predictions, _ = torch.mode(predictions, 1)
                samples_total += labels.size(0)
                correct_total += (predictions == labels).sum().item()
                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), predictions[i].item()] += 1

        # calculating mean accuracy and mean loss of the test set
        model_accuracy = correct_total / samples_total
        loss_total = loss_total / samples_total

        return model_accuracy, confusion_matrix, loss_total

    def test(self):
        test_accuracy, confusion_matrix, test_loss = self.calculate_accuracy_and_loss(data_type='test')
        print(f"test accuracy: {100 * test_accuracy:.4f}%",
              f"test loss: {test_loss:.4f}")


if __name__ == "__main__":

    # loading training options and hyper-parameters
    with open("options.json", 'r') as fp:
        options = json.load(fp)

    print(f"Starting test: {options['test_name']}")

    # tensorboard initialising
    tensorboard_path = os.path.join(options['tensorboard_dir'], options['test_name'])
    options['writer'] = SummaryWriter(log_dir=tensorboard_path)

    # check if need to load check points
    ckpt_dir = os.path.join(options["ckpt_dir"], options['test_name'])
    options["ckpt_dir"] = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    options["ckpt"] = LoadCkpt(ckpt_dir)

    # train
    train = Env(options=options, **options)
    Env.calculate_accuracy_and_loss('val')

    print("done training! Deep Learning Rules!")
