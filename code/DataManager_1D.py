import torchaudio
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn.functional as F
import random

class GTZANDataset(Dataset):
    def __init__(self, torch_dataset, labels_list, vector_equlizer='padding',
                 mode="sample", parts=12, sample_rate=22050,
                 output_length=675804, augmentation=False):
        self.mode = mode
        self.parts = parts
        self.output_length = output_length
        assert output_length % parts == 0
        self.part_length = output_length // parts
        self.sample_rate = sample_rate
        self.transforms = None
        if augmentation:
            self.transforms = get_augmentations()
        self.generator = None
        x = []
        y = []
        for item in torch_dataset:
            waveform, sr, label = item
            if vector_equlizer == 'padding':
                waveform = waveform
                pad = output_length - waveform.size(1)
                x.append(F.pad(input=waveform, pad=(0, pad, 0, 0), mode='constant', value=0))
                y.append(labels_list.index(label))
            elif vector_equlizer == 'cut min':
                x.append(waveform[:, :output_length])
                y.append(labels_list.index(label))
            elif vector_equlizer == 'k sec':
                k = k
                sec_k = sr * k
                resize_data_factor = 1
                r_list = createRandomSortedList(resize_data_factor, (0 * sr), (26 * sr))
                for i in r_list:
                    y.append(labels_list.index(label))
                    if i < 0:
                        pad = i * (-1)
                        x.append(F.pad(input=waveform[:, :i + sec_k], pad=(pad, 0, 0, 0), mode='constant', value=0))
                    elif i > sr * 25:
                        pad = i + sec_k - waveform.size(1)
                        x.append(F.pad(input=waveform[:, i:], pad=(0, pad, 0, 0), mode='constant', value=0))
                    else:
                        x.append(waveform[:, i:i + sec_k])
        self.x = torch.stack(x)
        self.y = torch.tensor(y)

    def __getitem__(self, index):
        if self.mode == "sample":  # sample one sample part
            start = random.randrange(self.output_length - self.part_length - 1)
            x, y = self.x[index][:, start:start + int(self.part_length)], self.y[index]
        elif self.mode == "split":  # return a regular sample
            x, y = self.x[index], self.y[index]
        else:
            raise Exception("mode parameter is not one of following: sample, split")
        if self.transforms is not None:
            x, y = self.transforms(x), y
        return x, y

    def __len__(self):
        return self.x.shape[0]

    def sample_k(self):
        pass

    def split_to_k(self):
        pass


def createRandomSortedList(num, start=1, end=100):
    arr = []
    tmp = random.randint(start, end)

    for x in range(num):

        while tmp in arr:
            tmp = random.randint(start, end)

        arr.append(tmp)

    arr.sort()

    return arr


def get_dataloader(hparams):

    trainset,validset,testset = load_gtza_from_torch()

    trainset = GTZANDataset(torch_dataset=trainset, labels_list=hparams.genres,vector_equlizer='k sec')
    validset = GTZANDataset(torch_dataset=validset, labels_list=hparams.genres,vector_equlizer='k sec')
    testset = GTZANDataset(torch_dataset=testset, labels_list=hparams.genres,vector_equlizer='k sec')

    train_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader


def load_gtza_from_torch():

    trainset = torchaudio.datasets.GTZAN(root="./datasets", download=True,subset="training")
    validset = torchaudio.datasets.GTZAN(root="./datasets", download=True, subset="validation")
    testset = torchaudio.datasets.GTZAN(root="./datasets", download=True, subset="testing")
    return trainset, validset, testset

def get_augmentations():
    import torchaudio_augmentations as taa
    sr = 22050
    augmentations = taa.Compose([
        taa.RandomApply([taa.Noise(min_snr=0.001, max_snr=0.01)], p=0.3),
        taa.RandomApply([taa.Gain()], p=0.2),
        # taa.RandomApply([taa.HighLowPass(sample_rate=sr)], p=0.2), # this augmentation will always be applied in this aumgentation chain!
        taa.RandomApply([taa.Delay(sample_rate=sr)], p=0.2),
    ])
    return augmentations

if __name__ == '__main__':
    load_gtza_from_torch()