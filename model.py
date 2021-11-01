import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechResModel(nn.Module):
    def __init__(self, n_labels, n_feature_maps=45, n_layers=26, dilation=True, res_pool=2):
        super().__init__()
        self.n_labels = n_labels
        self.n_maps = n_feature_maps
        self.conv0 = nn.Conv2d(1, self.n_maps, (3, 3), padding=(1, 1), bias=False)
        self.avg_pool = res_pool
        if res_pool:
            self.pool = nn.AvgPool2d(res_pool)

        self.n_layers = n_layers
        if dilation:
            self.convs = [nn.Conv2d(self.n_maps, self.n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(self.n_maps, self.n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(self.n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(self.n_maps, self.n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if self.avg_pool:
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)
