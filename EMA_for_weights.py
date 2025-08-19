import copy
import torch

class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = float(decay)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            if k in msd and esd[k].dtype == msd[k].dtype:
                esd[k].copy_(esd[k] * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.ema_model.state_dict()