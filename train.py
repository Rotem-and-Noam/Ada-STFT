import json
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from check_points import *
from get_components import *


class Train:

    def __init__(self, batch_size, num_workers, epoch_num, learning_rate, gamma, writer,
                 data_dir, ckpt, **kwargs):
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']
        self.train_data = get_dataloader(mode='train', data_dir=data_dir, genres=self.genres,
                                         batch_size=batch_size, num_workers=num_workers)
        self.val_data = get_dataloader(mode='val', data_dir=data_dir, genres=self.genres,
                                       batch_size=batch_size, num_workers=num_workers)
        self.data_length = len(self.train_data)
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(self.device, ckpt)
        self.optimizer, self.scheduler = get_optimizer(self.model, learning_rate, gamma, ckpt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epoch_num = epoch_num
        self.start_epoch = ckpt.start_epoch
        self.class_number = len(self.genres)
        self.ckpt = ckpt

    def train(self):
        for epoch in range(self.start_epoch, self.epoch_num):

            train_loss = self.train_epoch()
            val_accuracy, confusion_matrix, val_loss = self.calculate_accuracy_and_loss()

            print(f"epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
                  f"train loss: {train_loss:.4f}",
                  f"val loss: {val_loss:.4f}")

            # send documentation to tensorboard
            self.tensorboard_logging(confusion_matrix, train_loss, val_loss, val_accuracy, epoch)
            # save check points
            self.ckpt.save_ckpt(self.model, self.optimizer, self.scheduler, epoch)

    def tensorboard_logging(self, confusion_matrix, train_loss, val_loss, val_accuracy, epoch):
        self.writer.add_figure('confusion matrix', self.show_confusion_matrix(confusion_matrix))
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    def show_confusion_matrix(self, confusion_matrix, show=True):
        if show:
            plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(self.class_number), self.genres)
        plt.xlabel('Predicted Category')
        plt.xticks(range(self.class_number), self.genres)
        if show:
            plt.show()
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

    def calculate_accuracy_and_loss(self):
        self.model.eval()
        correct_total = 0
        samples_total = 0
        loss_total = 0
        confusion_matrix = np.zeros([self.class_number, self.class_number], int)

        # iterate on the test set and calculate accuracy and loss per batch
        with torch.no_grad():
            for images, labels in self.val_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item() * labels.size(0)
                _, predictions = torch.max(outputs.data, 1)
                samples_total += labels.size(0)
                correct_total += (predictions == labels).sum().item()
                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), predictions[i].item()] += 1

        # calculating mean accuracy and mean loss of the test set
        model_accuracy = correct_total / samples_total
        loss_total = loss_total / samples_total

        return model_accuracy, confusion_matrix, loss_total


if __name__ == "__main__":

    # loading training options and hyper-parameters
    with open("options.json", 'r') as fp:
        options = json.load(fp)

    # tensorboard initialising
    tensorboard_path = os.path.join(options['tensorboard_dir'], options['test_name'])
    options['writer'] = SummaryWriter(log_dir=tensorboard_path)

    # check if need to load check points
    ckpt_dir = os.path.join(options["ckpt_dir"], options['test_name'])
    options["ckpt_dir"] = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    options["ckpt"] = LoadCkpt(ckpt_dir)

    # train
    train = Train(**options)
    train.train()

    print("done training! Deep Learning Rules!")
