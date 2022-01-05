import os

import torch
import numpy as np
import torchaudio

from DataManager_1D import GTZANDataset
from model import Classifier
from get_components import *




def test(classifier, criterion, device, batch_size, num_workers, genres, data_dir):

    test_data = get_dataloader(mode='test', data_dir=data_dir, genres=genres,
                                batch_size=batch_size, num_workers=num_workers)
    test_accuracy, confusion_matrix, test_loss = calculate_accuracy_and_loss(classifier,
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
