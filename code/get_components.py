import torchaudio
import torch
from DataManager_1D import GTZANDataset
from code.model import Classifier


def get_augmentations():
    import torchaudio_augmentations as taa
    sr = 22050
    augmentations = taa.Compose([
        taa.RandomApply([taa.Noise(min_snr=0.001, max_snr=0.01)], p=0.3),
        taa.RandomApply([taa.Gain()], p=0.2),
        # taa.RandomApply([taa.HighLowPass(sample_rate=sr)], p=0.2), # this augmentation will always be applied in this aumgentation chain!
        taa.RandomApply([taa.Delay(sample_rate=sr)], p=0.2),
    ])


def get_dataloader(mode, data_dir, genres, batch_size, num_workers, parts):
    if mode == 'train':
        train_set = torchaudio.datasets.GTZAN(data_dir, subset="training", download=True)
        train_set = GTZANDataset(torch_dataset=train_set, labels_list=genres, vector_equlizer='padding',
                                 parts=parts)
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
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers)
        return val_data

    elif mode == 'test':
        test_set = torchaudio.datasets.GTZAN(data_dir, subset="testing", download=True)
        test_set = GTZANDataset(torch_dataset=test_set, labels_list=genres, vector_equlizer='padding',
                                mode="split", parts=parts)
        test_data = torch.utils.data.DataLoader(test_set,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
        return test_data
    else:
        raise Exception("data_type parameter is not one of following: val, test, train")


def get_model(device, ckpt, load_resnet_path=None):
    model = Classifier().to(device)
    if ckpt.is_ckpt():
        model = ckpt.load_model(model)
    elif load_resnet_path is not None:
        model.load_resnet_weights(load_resnet_path)
        model.to(device)
    return model


def get_optimizer(model, learning_rate, gamma, ckpt):
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    if ckpt.is_ckpt():
        ckpt.load_optimizer(optimizer)
        scheduler = ckpt.load_scheduler()
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return optimizer, scheduler

