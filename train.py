import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from DataManager_1D import GTZANDataset
from model import Classifier
from test import calculate_accuracy



genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']


def train(classifier, criterion, device, batch_size=4, num_workers=2, epoch_num=2, learning_rate=4e-2, gamma=0.9997):
    train_set = torchaudio.datasets.GTZAN(r"C:\Users\elata\code\MusicGenreClassifier\datasets", subset="training")
    train_set = GTZANDataset(torch_dataset=train_set, labels_list=genres, vector_equlizer='k sec')
    train_data = torch.utils.data.DataLoader(train_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    val_set = torchaudio.datasets.GTZAN(r"C:\Users\elata\code\MusicGenreClassifier\datasets", subset="validation")
    val_set = GTZANDataset(torch_dataset=val_set, labels_list=genres, vector_equlizer='k sec')
    val_data = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    length = len(train_set)

    optimizer = torch.optim.AdamW(classifier.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(epoch_num):
        train_loss = train_epoch(classifier, train_data, criterion, optimizer, device, length//batch_size)
        val_accuracy, confusion_matrix, val_loss = calculate_accuracy(model=classifier, dataloader=val_data,
                           device=device, criterion=criterion)
        print(f"epoch #{epoch}, val accuracy: {100 * val_accuracy:.4f}%",
              f"train loss: {train_loss:.4f}",
              f"val loss: {val_loss:.4f}")

        scheduler.step()


def train_epoch(classifier, train_data, criterion, optimizer, device, length):
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
    return train_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier()
    criterion = torch.nn.CrossEntropyLoss()
    train(classifier, criterion, device)
    torch.save(classifier.state_dict(), path=os.path.join(".", "saved_models", "classifier.torch"))
