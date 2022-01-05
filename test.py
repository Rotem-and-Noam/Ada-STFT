import os

import torch
import numpy as np
import torchaudio

from DataManager_1D import GTZANDataset
from model import Classifier
from train import genres


def calculate_accuracy(model, dataloader, device, criterion, class_number=10):
    model.eval()
    correct_total = 0
    samples_total = 0
    loss_total = 0
    confusion_matrix = np.zeros([class_number, class_number], int)

    # iterate on the test set and calculate accuracy and loss per batch
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
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


def test(classifier, criterion, device, batch_size=8, num_workers=2):

    test_set = torchaudio.datasets.GTZAN(r"C:\Users\elata\code\MusicGenreClassifier\datasets\genres", subset="test")
    test_set = GTZANDataset(torch_dataset=test_set, labels_list=genres, vector_equlizer='k sec')
    test_data = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    test_accuracy, confusion_matrix, test_loss = calculate_accuracy(classifier,
                                                                    test_data,
                                                                    device,
                                                                    criterion,
                                                                    class_number=10)
    print(f"test accuracy: {100 * test_accuracy:.4f}%",
          f"test loss: {test_loss:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier()
    criterion = torch.nn.CrossEntropyLoss()
    classifier.load_state_dict(torch.load(path=os.path.join(".", "saved_models", "classifier.torch")))
    test(classifier, criterion, device)
