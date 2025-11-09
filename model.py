import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

L1 = 512
L2 = 8
L3 = 64

class NNUE(pl.LightningModule):
    def __init__(self, feature_set, lambda_=[1.0], lr=[1.0],
                 label_smoothing_eps=0.0, num_epochs_to_adjust_lr=50,
                 score_scaling=361, momentum=0.0,
                 ply_begin_threshold=100.0, ply_end_threshold=120.0):
        super().__init__()
        self.input = nn.Linear(feature_set.num_features, L1)
        self.feature_set = feature_set
        self.l1 = nn.Linear(2*L1, L2)
        self.l2 = nn.Linear(L2, L3)
        self.output = nn.Linear(L3, 1)

        self.lambda_ = lambda_
        self.lr = lr
        self.label_smoothing_eps = label_smoothing_eps
        self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
        self.score_scaling = score_scaling
        self.momentum = momentum
        self.ply_begin_threshold = ply_begin_threshold
        self.ply_end_threshold = ply_end_threshold

        self._zero_virtual_feature_weights()
        self._scheduler = None  # 後で設定

    def _zero_virtual_feature_weights(self):
        weights = self.input.weight
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                weights[:, a:b] = 0.0
        self.input.weight = nn.Parameter(weights)

    def forward(self, us, them, w_in, b_in):
        w = self.input(w_in)
        b = self.input(b_in)
        l0_ = (us * torch.cat([w,b], dim=1)) + (them * torch.cat([b,w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
        l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
        x = self.output(l2_)
        return x

    def step_(self, batch, batch_idx, loss_type):
        us, them, white, black, outcome, score, ply = batch
        nnue2score = 600
        scaling = self.score_scaling
        q = self(us, them, white, black) * nnue2score / scaling
        t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
        p = (score / scaling).sigmoid()

        eps = 1e-12
        teacher_entropy = -(p * (p+eps).log() + (1-p)*(1-p+eps).log())
        outcome_entropy = -(t*(t+eps).log() + (1-t)*(1-t+eps).log())
        teacher_loss = -(p*F.logsigmoid(q) + (1-p)*F.logsigmoid(-q))
        outcome_loss = -(t*F.logsigmoid(q) + (1-t)*F.logsigmoid(-q))

        lambda_ = self.lambda_[0] if self.lambda_[0] >= 0.0 else torch.clamp(
            (self.ply_end_threshold - ply) / (self.ply_end_threshold - self.ply_begin_threshold), 0.0, 1.0
        )

        result = lambda_*teacher_loss + (1-lambda_)*outcome_loss
        entropy = lambda_*teacher_entropy + (1-lambda_)*outcome_entropy
        loss = result.mean() - entropy.mean()
        self.log(loss_type, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, 'val_loss')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # 標準更新
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        # ミニバッチ単位で scheduler を更新
        if self._scheduler is not None:
            self._scheduler.step(epoch + batch_idx / self.trainer.num_training_batches)
            self.log("lr", self._scheduler.get_last_lr()[0])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr[0],
            momentum=self.momentum,
            weight_decay=0.1
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.num_epochs_to_adjust_lr,
            T_mult=1,
            eta_min=1e-5
        )
        self._scheduler = scheduler
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def get_layers(self, filt):
        for i in self.children():
            if filt(i) and isinstance(i, nn.Linear):
                for p in i.parameters():
                    if p.requires_grad:
                        yield p
