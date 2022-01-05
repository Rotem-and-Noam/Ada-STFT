import json
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from DataManager_1D import GTZANDataset
from model import Classifier
from test import calculate_accuracy_and_loss
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']


def train(classifier, criterion, device, batch_size, num_workers, epoch_num, learning_rate, gamma, writer, data_dir, **kwargs):
    train_set = torchaudio.datasets.GTZAN(data_dir, subset="training", download=True)
    train_set = GTZANDataset(torch_dataset=train_set, labels_list=genres, vector_equlizer='k sec')
    train_data = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             drop_last=True)
    val_set = torchaudio.datasets.GTZAN(data_dir, subset="validation", download=True)
    val_set = GTZANDataset(torch_dataset=val_set, labels_list=genres, vector_equlizer='k sec')
    val_data = torch.utils.data.DataLoader(val_set,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           drop_last=True)
    length = len(train_set)

    optimizer = torch.optim.AdamW(classifier.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(epoch_num):
        train_loss = train_epoch(classifier, train_data, criterion, optimizer, scheduler, device, length//batch_size)
        val_accuracy, confusion_matrix, val_loss = calculate_accuracy_and_loss(model=classifier, dataloader=val_data,
                                                                               device=device, criterion=criterion)

        print(f"epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
              f"train loss: {train_loss:.4f}",
              f"val loss: {val_loss:.4f}")

        writer.add_figure('confusion matrix', show_confusion_matrix(confusion_matrix))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)



def show_confusion_matrix(confusion_matrix, show=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(10), genres)
    plt.xlabel('Predicted Category')
    plt.xticks(range(10), genres)
    if show:
        plt.show()
    return fig


def train_epoch(classifier, train_data, criterion, optimizer, scheduler, device, length):
    classifier.train()
    train_epoch_loss = 0
    samples_total = 0
    with tqdm(total=length) as pbar:
        for j, sample in enumerate(train_data):
            waveforms, labels = sample

            # getting batch output and calculating batch loss
            output = classifier(waveforms.to(device))
            loss = criterion(output, labels.to(device))

            # the three musketeers:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updating parameters for calcultaing total loss
            train_epoch_loss += loss.detach().item() * labels.size(0)
            samples_total += labels.size(0)
            pbar.update()

    # calculating mean train loss for epoch
    train_loss = train_epoch_loss / samples_total
    scheduler.step()

    return train_loss


if __name__ == "__main__":

    with open("options.json", 'r') as fp:
        options = json.load(fp)

    tensorboard_path = os.path.join(options['tensorboard_dir'], options['test_name'])
    writer = SummaryWriter(log_dir=tensorboard_path)
    options['writer'] = writer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier()
    criterion = torch.nn.CrossEntropyLoss()
    train(classifier, criterion, device, **options)
    torch.save(classifier.state_dict(), path=os.path.join(".", "saved_models", "classifier.torch"))
