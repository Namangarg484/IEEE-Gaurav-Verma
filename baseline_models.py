"""
baseline_models.py
==================
Re-implementations of published single-channel PPG regressors, evaluated in the
same subject-wise harness as the fusion network (run_subject_cv.py).  All take
the (x1, x2, x3) call signature of MultimodalFusionNet and use only x1.

ResNet1D  : stem + 3 residual blocks (64-128-256), GAP, MLP head  (~1.1 M params)
CNNLSTM   : 3 conv layers (32-64-128) -> 2-layer LSTM(128) -> MLP head
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class _ResBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(out + idt)


class ResNet1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(64),
                                  nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
        self.layers = nn.Sequential(_ResBlock1D(64, 64), _ResBlock1D(64, 128, 2), _ResBlock1D(128, 256, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                  nn.Linear(64, 1))
        _init(self)

    def forward(self, x1, x2=None, x3=None):
        return self.head(self.pool(self.layers(self.stem(x1))).squeeze(-1))


class CNNLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                  nn.Linear(64, 1))
        _init(self)

    def forward(self, x1, x2=None, x3=None):
        _, (h, _) = self.lstm(self.cnn(x1).permute(0, 2, 1))
        return self.head(h[-1])


BASELINES = {"resnet1d": ResNet1D, "cnnlstm": CNNLSTM}


if __name__ == "__main__":
    for k, cls in BASELINES.items():
        m = cls(); out = m(torch.randn(4, 1, 320))
        print(k, tuple(out.shape), f"{sum(p.numel() for p in m.parameters()):,} params")
