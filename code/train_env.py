import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from check_points import *
from get_components import *
from options_parser import get_options


class Env:

    def __init__(self, batch_size, num_workers, epoch_num, learning_rate, gamma, writer=None,
                 data_dir, ckpt, ckpt_interval, options, load_resnet_weight_path=None, optimizer_class="AdamW",
                 split_parts=1, learn_window=0, learn_kernels=0, cpu=False, augmentation=False, three_windows=0,
                 test_name=None, **kwargs):
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']
        self.train_data = get_dataloader(mode='train', data_dir=data_dir, genres=self.genres,
                                         batch_size=batch_size, num_workers=num_workers, parts=split_parts,
                                         augmentation=augmentation)
        self.val_data = get_dataloader(mode='val', data_dir=data_dir, genres=self.genres,
                                       batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.test_data = get_dataloader(mode='test', data_dir=data_dir, genres=self.genres,
                                        batch_size=batch_size, num_workers=num_workers, parts=split_parts)
        self.batch_size = batch_size
        self.data_length = len(self.train_data)
        self.writer = writer
        if not cpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = get_model(self.device, ckpt, three_windows, load_resnet_weight_path)
        self.optimizer, self.scheduler = get_optimizer(self.model, learning_rate, gamma, ckpt, optimizer_class)
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
        self.three_windows = three_windows
        self.test_name = test_name
        print(f"Learnable parameters {self.count_parameters(self.model)}")
        print(f"Starting training on device {str(self.device)} for {str(self.epoch_num)} epochs")

    def train(self):
        for epoch in range(self.start_epoch, self.epoch_num):

            train_loss = self.train_epoch()
            val_accuracy, confusion_matrix, val_loss = self.calculate_accuracy_and_loss()

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
            if self.three_windows:
                self.writer.add_figure('Window2',
                                       self.show_window(self.model.stft1.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                       epoch)
                self.writer.add_figure('Window3',
                                       self.show_window(self.model.stft2.win_cof.detach().clone().squeeze(0).cpu().numpy()),
                                       epoch)

    def show_confusion_matrix(self, confusion_matrix, show=False):
        if show:
            plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0,
                   vmax=len(self.val_data)/self.class_number, cmap=plt.get_cmap('Blues'))
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
                image = images.to(self.device)
                labels = labels.to(self.device)
                image_splited = image.reshape(image.shape[0] * self.split_parts, 1, -1)
                label_splited = labels.repeat(self.split_parts, 1).T.reshape(-1)
                outputs = self.model(image_splited)
                loss = self.criterion(outputs, label_splited)
                loss_total += loss.item() * len(labels)
                _, predictions = torch.max(outputs.data, 1)
                prediction, _ = torch.mode(predictions)
                samples_total += labels.size(0)
                correct_total += (prediction == labels).sum().item()
                confusion_matrix[labels.item(), prediction.item()] += 1

        # calculating mean accuracy and mean loss of the test set
        model_accuracy = correct_total / samples_total
        loss_total = loss_total / samples_total

        return model_accuracy, confusion_matrix, loss_total

    def test(self):
        test_accuracy, confusion_matrix, test_loss = self.calculate_accuracy_and_loss(data_type='test')
        print(f"test accuracy: {100 * test_accuracy:.4f}%",
              f"test loss: {test_loss:.4f}")

def check_checkpoints(options):
    ckpt_dir = os.path.join(options["ckpt_dir"], options['test_name'])
    options["ckpt_dir"] = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = LoadCkpt(ckpt_dir)
    return ckpt

if __name__ == "__main__":

    # loading training options and hyper-parameters
    parser = get_options()
    options = vars(parser.parse_args())

    print(f"Starting test: {options['test_name']}")

    # check if need to load check points
    ckpt = check_checkpoints(options)

    if ckpt.start_epoch >= options['epoch_num']:
        print('This test is already done!')

    else:
        # tensorboard initialising
        tensorboard_path = os.path.join(options['tensorboard_dir'], options['test_name'])
        writer = SummaryWriter(log_dir=tensorboard_path)

        # train
        env = Env(writer=writer, ckpt=ckpt, options=options, **options)
        # env.calculate_accuracy_and_loss('val')
        env.train()

        print("done training! Deep Learning Rules!")
