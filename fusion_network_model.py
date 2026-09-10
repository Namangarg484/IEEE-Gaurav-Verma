"""
fusion_network_model.py
=======================
Multimodal 1D+2D+3D Deep Fusion Network for Non-Invasive Blood Glucose Prediction.

Architecture
------------
    Raw PPG Signal (B, 1, 500)
         │
    ┌────┼──────────────────────────┐
    │    │                          │
 Branch1D     Branch2D          Branch3D
 1D-CNN       2D-CNN (CWT)      PointNet (Phase-Space)
 → R^256      → R^256           → R^256
    │    │                          │
    └────┴──────────────────────────┘
          CrossAttentionFusion
          → R^768  (three weighted branches)
          │
      FC Head: 768→256→64→1
          │
      Glucose (mg/dL)

Key design decisions
--------------------
- Equal 256-dim embeddings from all branches  → balanced fusion (no branch dominates).
- BatchNorm after every convolution            → training stability, less oscillation.
- Kaiming weight initialisation               → proper gradient flow from epoch 1.
- Scaled dot-product attention                → avoids score saturation vs raw dot-product.
- Gradient clipping enforced in trainer       → further guards against instability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Branch 1 — Temporal Morphology (1D-CNN)
# ---------------------------------------------------------------------------
class Branch1D(nn.Module):
    """1D-CNN on raw Z-score-normalised PPG.  Input: (B,1,500)  Output: (B,256)"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        # Global average pool (output_size=1 always divides on MPS)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.proj(x)          # (B, 256)


# ---------------------------------------------------------------------------
# Branch 2 — Time-Frequency Domain (2D-CNN on CWT Scalogram)
# ---------------------------------------------------------------------------
class Branch2D(nn.Module):
    """2D-CNN on CWT scalogram (64x64 image).  Input: (B,1,64,64)  Output: (B,256)"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # → (B,32,32,32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # → (B,64,16,16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # → (B,128,8,8)
        )
        # Global average pool (output_size=1 always divides on MPS)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.proj(x)          # (B, 256)


# ---------------------------------------------------------------------------
# Branch 3 — Non-linear Dynamics (PointNet-style on 3D Phase Space)
# ---------------------------------------------------------------------------
class Branch3D(nn.Module):
    """PointNet MLP on Takens delay-embedding point cloud.
    Input: (B,3,256)  Output: (B,256)"""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3,   64,  kernel_size=1, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Conv1d(64,  128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)                       # (B, 256, N)
        x = torch.max(x, dim=2).values        # global max-pool → (B, 256)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Block
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """Scaled learnable-query attention over three branch embeddings.

    For f1, f2, f3 ∈ R^d each branch is projected to R^att_dim,
    scored against a learnable query q, softmax-normalised, and the
    resulting weights are applied to the original branch features.

    Input : list of 3 tensors (B, branch_dim)
    Output: concatenation of weighted features (B, 3*branch_dim) = (B, 768)
    """

    def __init__(self, branch_dim: int = 256, att_dim: int = 128) -> None:
        super().__init__()
        self.att_dim = att_dim
        self.projs   = nn.ModuleList([
            nn.Linear(branch_dim, att_dim, bias=False) for _ in range(3)
        ])
        # Small initialisation prevents one branch dominating from the start
        self.query = nn.Parameter(torch.randn(att_dim) * 0.02)

    def forward(self, feats: list) -> torch.Tensor:
        # Stack projected features → (B, 3, att_dim)
        stacked = torch.stack(
            [proj(f) for proj, f in zip(self.projs, feats)], dim=1
        )
        # Scaled dot-product with learnable query → (B, 3, 1)
        q      = self.query.view(1, self.att_dim, 1)
        scores = torch.bmm(stacked, q.expand(stacked.size(0), -1, -1))
        scores = scores / (self.att_dim ** 0.5)
        weights = F.softmax(scores, dim=1)     # (B, 3, 1)

        # Apply weights to ORIGINAL branch features (not projected)
        weighted = [feats[i] * weights[:, i, :] for i in range(3)]
        return torch.cat(weighted, dim=1)      # (B, 768)

    def get_weights(self, feats: list) -> torch.Tensor:
        """Return per-branch attention weights (B, 3) for interpretability."""
        stacked = torch.stack(
            [proj(f) for proj, f in zip(self.projs, feats)], dim=1
        )
        q      = self.query.view(1, self.att_dim, 1)
        scores = torch.bmm(stacked, q.expand(stacked.size(0), -1, -1))
        scores = scores / (self.att_dim ** 0.5)
        return F.softmax(scores, dim=1).squeeze(-1)  # (B, 3)


# ---------------------------------------------------------------------------
# Full Multimodal Fusion Network
# ---------------------------------------------------------------------------
class MultimodalFusionNet(nn.Module):
    """Complete 1D-2D-3D fusion network.

    Inputs:
        input_1d : (B, 1,    500)  – raw PPG windows
        input_2d : (B, 1,  64,64)  – CWT scalogram
        input_3d : (B, 3,    256)  – phase-space point cloud

    Output:
        (B, 1) – predicted blood glucose in mg/dL
    """

    BRANCH_DIM = 256
    FUSED_DIM  = BRANCH_DIM * 3   # 768

    def __init__(self) -> None:
        super().__init__()
        self.branch_1d = Branch1D()
        self.branch_2d = Branch2D()
        self.branch_3d = Branch3D()
        self.fusion    = CrossAttentionFusion(self.BRANCH_DIM)

        self.head = nn.Sequential(
            nn.Linear(self.FUSED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        input_3d: torch.Tensor,
    ) -> torch.Tensor:
        f1 = self.branch_1d(input_1d)
        f2 = self.branch_2d(input_2d)
        f3 = self.branch_3d(input_3d)
        fused = self.fusion([f1, f2, f3])
        return self.head(fused)

    def forward_with_attention(
        self,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        input_3d: torch.Tensor,
    ) -> tuple:
        """Returns (predictions, attention_weights) for interpretability."""
        f1 = self.branch_1d(input_1d)
        f2 = self.branch_2d(input_2d)
        f3 = self.branch_3d(input_3d)
        weights = self.fusion.get_weights([f1, f2, f3])  # (B, 3)
        fused   = self.fusion([f1, f2, f3])
        return self.head(fused), weights

    def forward_with_latent(
        self,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        input_3d: torch.Tensor,
    ) -> tuple:
        """Returns (predictions, latent_768d, attention_weights).
        latent_768d is the fused representation before the FC head.
        Used for t-SNE and representation analysis.
        """
        f1 = self.branch_1d(input_1d)
        f2 = self.branch_2d(input_2d)
        f3 = self.branch_3d(input_3d)
        weights = self.fusion.get_weights([f1, f2, f3])  # (B, 3)
        fused   = self.fusion([f1, f2, f3])               # (B, 768)
        return self.head(fused), fused, weights


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = MultimodalFusionNet()
    B = 4
    d1 = torch.randn(B, 1, 500)
    d2 = torch.randn(B, 1, 64, 64)
    d3 = torch.randn(B, 3, 256)

    f1 = model.branch_1d(d1)
    f2 = model.branch_2d(d2)
    f3 = model.branch_3d(d3)
    out, w = model.forward_with_attention(d1, d2, d3)

    assert f1.shape == (B, 256), f"Branch1D failed: {f1.shape}"
    assert f2.shape == (B, 256), f"Branch2D failed: {f2.shape}"
    assert f3.shape == (B, 256), f"Branch3D failed: {f3.shape}"
    assert out.shape == (B, 1),  f"Output failed:   {out.shape}"
    assert w.shape   == (B, 3),  f"Weights failed:  {w.shape}"

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Branch1D : {f1.shape}")
    print(f"Branch2D : {f2.shape}")
    print(f"Branch3D : {f3.shape}")
    print(f"Output   : {out.shape}")
    print(f"Attention: {w[0].tolist()}")
    print(f"Parameters: {total:,}")
    print("fusion_network_model.py  OK")
