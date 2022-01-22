import torchaudio
import torch
from data_utils.DataManager_1D import GTZANDataset
from models.model import Classifier


def get_dataloader(mode, data_dir, genres, batch_size, num_workers, parts, augmentation=False):
    if mode == 'train':
        train_set = torchaudio.datasets.GTZAN(data_dir, subset="training", download=True)
        train_set = GTZANDataset(torch_dataset=train_set, labels_list=genres, vector_equlizer='padding',
                                 parts=parts, augmentation=augmentation)
        train_data = torch.utils.data.DataLoader(train_set,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 drop_last=True)
        return train_data

    elif mode == 'val':
        val_set = torchaudio.datasets.GTZAN(data_dir, subset="validation", download=True)
        val_set = GTZANDataset(torch_dataset=val_set, labels_list=genres, vector_equlizer='padding',
                               mode="split", parts=parts)
        val_data = torch.utils.data.DataLoader(val_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=num_workers)
        return val_data

    elif mode == 'test':
        test_set = torchaudio.datasets.GTZAN(data_dir, subset="testing", download=True)
        test_set = GTZANDataset(torch_dataset=test_set, labels_list=genres, vector_equlizer='padding',
                                mode="split", parts=parts)
        test_data = torch.utils.data.DataLoader(test_set,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=num_workers)
        return test_data
    else:
        raise Exception("data_type parameter is not one of following: val, test, train")


def get_model(device, ckpt, three_windows, load_resnet_path=None):
    model = Classifier(three_windows=three_windows).to(device)
    if ckpt.is_ckpt():
        model = ckpt.load_model(model)
    elif load_resnet_path is not None:
        model.load_resnet_weights(load_resnet_path)
        model.to(device)
    return model


def get_optimizer(model, learning_rate, gamma, ckpt, optimizer_class):
    if optimizer_class == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_class == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if ckpt.is_ckpt():
        optimizer = ckpt.load_optimizer(optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = ckpt.load_scheduler(scheduler)

    return optimizer, scheduler

