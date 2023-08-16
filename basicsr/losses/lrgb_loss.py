import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


class Classification(nn.Module):
    def __init__(self, num_classes):
        super(Classification, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, num_classes, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        out = self.conv(x)
        return out.softmax(dim=1)


class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            )
            
    def forward(self, x):
        out = self.conv_stack(x)
        return out


class Decoder(nn.Module):
    def __init__(self, legacy=False):
        super(Decoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=legacy),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=legacy),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        return self.conv_stack(x)


class AutoEncoder(nn.Module):
    def __init__(self, num_classes=20, legacy=False):
        super(AutoEncoder, self).__init__()
        self.num_classes = num_classes
        self.classification = Classification(self.num_classes)
        self.encoder = nn.ModuleList()
        for i in range(self.num_classes):
            self.encoder.append(Encoder())
        self.decoder = Decoder(legacy)
        self.L1 = nn.L1Loss()

    def top1(self, t):
        values, index = t.topk(k=1, dim=1)
        values, index = map(lambda x: x.squeeze(dim=1), (values, index))
        return values, index

    def load_balancing_loss(self, routing):
        _, index = self.top1(routing)
        mask = F.one_hot(index, routing.shape[1]).float()
        mask = mask.reshape(mask.shape[0], -1, mask.shape[-1])
        density = mask.mean(dim=1)
        routing = routing.reshape(routing.shape[0], routing.shape[1], -1)
        density_proxy = routing.mean(dim=-1)
        balancing_loss = (density_proxy * density).mean() * float(routing.shape[1] ** 2)
        return balancing_loss

    def encode(self, x, get_classes=False):
        classes = self.classification(x)
        classes_top = torch.max(classes, dim=1)[1].unsqueeze(1).float()
        # init
        emb = self.encoder[0](x)
        for i in range(1, self.num_classes):
            emb = torch.where(classes_top == i, self.encoder[i](x), emb)
        if get_classes:
            return emb, classes
        return emb

    def decode(self, emb):
        return self.decoder(emb)

    def forward(self, x, loss_weight=0.1):
        """Training code for the autoencoder.

        The loss being used is `1 * (L1) + 0.1 * (Load Balancing Loss)`.
        L1 loss is calculated in the outside of the model.
        """
        emb, classes = self.encode(x, True)
        out = self.decode(emb + torch.randn_like(emb))
        
        load_balancing_loss = self.load_balancing_loss(classes)
        return out, load_balancing_loss * loss_weight


@LOSS_REGISTRY.register()
class LRGBLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0, 
        ae_n_experts: int = 20,
        ae_legacy: bool = False,
        checkpoint: str = '../pretrained/model_latest.pt',
        loss_type: str = 'L1',
        remap: dict = None,
    ) -> None:
        super(LRGBLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type.lower()
        self.remap = remap

        self.encoder = AutoEncoder(ae_n_experts, ae_legacy)
        self.encoder.load_state_dict(torch.load(checkpoint))
        for p in self.parameters():
            p.requires_grad = False

        if self.loss_type in ('l1', 'mae'):
            self.dist = F.l1_loss
        elif self.loss_type in ('l2', 'mse'):
            self.dist = F.mse_loss
        elif self.loss_type in ('huber', 'smoothl1', 'sl1'):
            self.dist = F.smooth_l1_loss
        else:
            raise ValueError(
                'Unknown loss type [{:s}] is detected'.format(loss_type))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Loss={self.loss_type})"

    @torch.cuda.amp.autocast(False)
    def forward(
        self,
        x: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        if self.remap is not None:
            x = kwargs[self.remap['x']]
            gt = kwargs[self.remap['gt']]
        x_feat = self._get_features(x)
        with torch.no_grad():
            gt_feat = self._get_features(gt)
        return self.loss_weight * self.dist(x_feat, gt_feat)

    def state_dict(self, *args, **kwargs):
        return {}

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)
