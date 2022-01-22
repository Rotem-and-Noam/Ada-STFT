import numpy as np
import matplotlib.pyplot as plt
import torch
from env_utils.get_components import *
from tqdm import tqdm


class Env:
    """"
    The entire deep learning environment of our training and testing.
    This class initiates all the needed parameters for test, train and val modes.
    Have functions to train, test and validate our data, and some visualizations functions:
    we use tensorboard and matplotlib in order to keep track on our training trials.
    """

    def __init__(self, batch_size, num_workers, epoch_num, learning_rate, gamma,
                 data_dir, ckpt, ckpt_interval, options, writer=None, load_resnet_weight_path=None,
                 split_parts=1, learn_window=0, learn_kernels=0, cpu=False, augmentation=False, three_windows=0,
                 optimizer_class="AdamW", test_name=None, **kwargs):

        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']

        # initialise data loaders
        self.train_data = get_dataloader(mode='train', data_dir=data_dir, genres=self.genres,
                                         batch_size=batch_size, num_workers=num_workers, parts=split_parts,
                                         augmentation=augmentation)
        self.val_data = get_dataloader(mode='val', data_dir=data_dir, genres=self.genres,
                                       batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.test_data = get_dataloader(mode='test', data_dir=data_dir, genres=self.genres,
                                        batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.data_length = len(self.train_data)

        # initialise tensorboard log dir
        self.writer = writer

        # initialise device
        if not cpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # initialise training modules: model's network, optimizer method and loss function (criterion)
        self.model = get_model(self.device, ckpt, three_windows, load_resnet_weight_path)
        self.optimizer, self.scheduler = get_optimizer(self.model, learning_rate, gamma, ckpt, optimizer_class)
        self.criterion = torch.nn.CrossEntropyLoss()

        # set hyper-parameters
        self.split_parts = split_parts
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.start_epoch = ckpt.start_epoch
        self.class_number = len(self.genres)
        self.options = options
        self.ckpt = ckpt
        self.ckpt_interval = ckpt_interval
        self.three_windows = three_windows

        # if we want to learn stft window or kernels DFT coefficients, set it
        if learn_window:
            self.model.stft.learn_window()
            if self.three_windows:
                self.model.stft1.learn_window()
                self.model.stft2.learn_window()
        if learn_kernels:
            self.model.stft.learn_kernels()
            if self.three_windows:
                self.model.stft1.learn_kernels()
                self.model.stft2.learn_kernels()
        self.model.stft.print_learnable_params()

        # report and set the trial's parameters
        self.test_name = test_name
        print(f"Learnable parameters {self.count_parameters(self.model)}")
        print(f"Starting training on device {str(self.device)} for {str(self.epoch_num)} epochs")
        self.best_acc = 0.5

    def train(self):
        for epoch in range(self.start_epoch, self.epoch_num):

            # train rpoch
            train_loss = self.train_epoch()
            # compute accuracy
            val_accuracy, confusion_matrix, val_loss = self.calculate_accuracy_and_loss()

            # keep track of the best accuracy so far
            if val_accuracy > self.best_acc:
                self.best_acc = val_accuracy
                self.ckpt.save_ckpt(self.model, self.optimizer, self.scheduler, epoch, self.options, True)
                if self.writer is not None:
                    self.writer.add_figure('best confusion matrix', self.show_confusion_matrix(confusion_matrix, val_accuracy),
                                           epoch)

            # report our current status
            print(f"{self.test_name}: epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
                  f"train loss: {train_loss:.5f}",
                  f"val loss: {val_loss:.5f}",
                  f"learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # send documentation to tensorboard
            if self.writer is not None:
                self.tensorboard_logging(confusion_matrix, train_loss, val_loss, val_accuracy, epoch)
            # save check points
            if epoch % self.ckpt_interval == self.ckpt_interval - 1:
                self.ckpt.save_ckpt(self.model, self.optimizer, self.scheduler, epoch, self.options)

    @staticmethod
    def count_parameters(model):
        # count our model's parameters
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def tensorboard_logging(self, confusion_matrix, train_loss, val_loss, val_accuracy, epoch):
        # save some values to be recorded in tensorboard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        self.writer.add_scalar('Window Change', self.model.stft.calc_window_change().item(), epoch)
        self.writer.add_scalar('Kernel Change', self.model.stft.calc_kernels_change().item(), epoch)
        if epoch % self.ckpt_interval == 0:
            self.writer.add_figure('confusion matrix', self.show_confusion_matrix(confusion_matrix, val_accuracy), epoch)
            self.writer.add_figure('Window',
                                   self.show_window(self.model.stft.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                   epoch)
            if self.three_windows:
                self.writer.add_figure('Window_2',
                                       self.show_window(self.model.stft1.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                       epoch)
                self.writer.add_figure('Window_3',
                                       self.show_window(self.model.stft2.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                       epoch)

    def show_confusion_matrix(self, confusion_matrix, accuracy, show=False):
        # create plot of confusion matrix
        if show:
            plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0,
                   vmax=len(self.val_data)/self.class_number, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(self.class_number), self.genres)
        plt.xlabel('Predicted Category')
        plt.xticks(range(self.class_number), self.genres)
        plt.title(f"Accuracy: {accuracy}")
        if show:
            plt.show()
        return fig

    @staticmethod
    def show_window(window):
        # show a figure of the window used in the stft computations
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

        # initialise parameters
        self.model.eval()
        correct_total = 0
        samples_total = 0
        loss_total = 0
        confusion_matrix = np.zeros([self.class_number, self.class_number], int)

        # pick the right data to compute accuracy for
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
                image = images.to(self.device)
                labels = labels.to(self.device)
                # split images
                image_split = image.reshape(image.shape[0] * self.split_parts, 1, -1)
                label_split = labels.repeat(self.split_parts, 1).T.reshape(-1)
                outputs = self.model(image_split)
                # compute loss
                loss = self.criterion(outputs, label_split)
                loss_total += loss.item() * len(labels)
                # compute predictions
                _, predictions = torch.max(outputs.data, 1)
                prediction, _ = torch.mode(predictions)
                samples_total += labels.size(0)
                # compute values for accuracy computations
                indicator = (prediction == labels)
                correct_total += torch.sum(indicator).item()
                confusion_matrix[labels.item(), prediction.item()] += 1

        # calculating mean accuracy and mean loss of the test set
        model_accuracy = correct_total / samples_total
        loss_total = loss_total / samples_total

        return model_accuracy, confusion_matrix, loss_total

    def test(self):
        test_accuracy, confusion_matrix, test_loss = self.calculate_accuracy_and_loss(data_type='test')
        print(f"test accuracy: {100 * test_accuracy:.4f}%",
              f"test loss: {test_loss:.4f}")
