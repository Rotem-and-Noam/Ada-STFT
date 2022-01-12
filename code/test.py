import os

import torch
import numpy as np
import torchaudio

from DataManager_1D import GTZANDataset
from model import Classifier
from get_components import *

genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']

def calculate_accuracy_and_loss(model, val_data, device, criterion, class_number, parts):
    model.eval()
    correct_total = 0
    samples_total = 0
    loss_total = 0
    confusion_matrix = np.zeros([class_number, class_number], int)

    # iterate on the test set and calculate accuracy and loss per batch
    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(device)
            labels_expanded = labels.to(device).repeat(parts, 1).T.reshape(-1)
            outputs = model(images)
            loss = criterion(outputs, labels_expanded)
            loss_total += loss.item() * len(labels)
            _, predictions = torch.max(outputs.data, 1)
            predictions = predictions.reshape(-1, parts)
            predictions, _ = torch.mode(predictions, 1)
            samples_total += labels.size(0)
            correct_total += (predictions == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predictions[i].item()] += 1

    # calculating mean accuracy and mean loss of the test set
    model_accuracy = correct_total / samples_total
    loss_total = loss_total / samples_total

    return model_accuracy, confusion_matrix, loss_total


def test(classifier, criterion, device, batch_size, num_workers, genres, data_dir, parts):

    test_data = get_dataloader(mode='test', data_dir=data_dir, genres=genres,
                                batch_size=batch_size, num_workers=num_workers)
    test_accuracy, confusion_matrix, test_loss = calculate_accuracy_and_loss(classifier,
                                                                             test_data,
                                                                             device,
                                                                             criterion,
                                                                             class_number=10,
                                                                             parts=12)
    print(f"test accuracy: {100 * test_accuracy:.4f}%",
          f"test loss: {test_loss:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    classifier = Classifier()
    criterion = torch.nn.CrossEntropyLoss()
    # classifier.load_state_dict(torch.load(path=os.path.join(".", "saved_models", "classifier.torch")))
    test(classifier, criterion, device, batch_size=16, num_workers=2, genres=genres, data_dir="../dataset", parts=12)
