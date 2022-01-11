import torch
import os


class LoadCkpt:

    def __init__(self, ckpt_dir):
        self._dir = ckpt_dir
        self._last_file = 'ckpt_last'
        self._ckpt_dict = self._load_ckpt()
        self.start_epoch = 0
        if not(self._ckpt_dict is None):
            self.start_epoch = self._load_start_epoch()

    def _load_ckpt(self):
        path = os.path.join(self._dir, self._last_file)
        if not os.path.isfile(path):
            return None
        else:
            return torch.load(path)

    def is_ckpt(self):
        if self._ckpt_dict is None:
            return False
        else:
            return True

    def load_model(self, model):
        _model = model
        _model.load_state_dict(self._ckpt_dict['state_dict'])
        return _model

    def load_optimizer(self, optimizer):
        _optimizer = optimizer
        optimizer.load_state_dict(self._ckpt_dict['optimizer_state_dict'])
        return _optimizer

    def _load_start_epoch(self):
        return self._ckpt_dict['epoch'] + 1

    def load_scheduler(self):
        return self._ckpt_dict['scheduler']

    def save_ckpt(self, model, optimizer, scheduler, epoch):
        ckpt_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler
        }

        torch.save(ckpt_dict, os.path.join(self._dir, f"ckpt_{epoch}.pt"))
        torch.save(ckpt_dict, os.path.join(self._dir, self._last_file))
