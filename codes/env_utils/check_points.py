import torch
import os


class LoadCkpt:

    def __init__(self, ckpt_dir, resume, ckpt_file, **kwargs):
        self._dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self._last_file = 'ckpt_last.pt'
        self._best_file = 'ckpt_best.pt'
        self._ckpt_file = ckpt_file
        self._ckpt_dict = None
        self.resume = resume
        if resume:
            self._ckpt_dict = self._load_ckpt()
        self.start_epoch = 0
        if self._ckpt_dict is not None:
            self.start_epoch = self._load_start_epoch()
            print(f"loaded check points, starting from epoch: {self.start_epoch}")

    def _load_ckpt(self):
        if self.resume:
            file = self._last_file
        else:
            file = self._ckpt_file
        path = os.path.join(self._dir, file)
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

    def load_scheduler(self, scheduler):
        _scheduler = scheduler
        _scheduler.load_state_dict(self._ckpt_dict['scheduler'].state_dict())
        return _scheduler

    def load_options(self):
        return self._ckpt_dict['options']

    def save_ckpt(self, model, optimizer, scheduler, epoch, options, best=False):
        ckpt_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler,
            'options': options
        }

        if not best:
            torch.save(ckpt_dict, os.path.join(self._dir, self._last_file))
        else:
            torch.save(ckpt_dict, os.path.join(self._dir, self._best_file))
