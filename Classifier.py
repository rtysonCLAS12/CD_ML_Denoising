import torch
from torch import jit
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import Callback
from typing import List

class EfficientGravNetLayer(nn.Module):
    def __init__(self, in_features, hidden_features, k=16, concat_input=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.s_dim = 1 #tried 50, doesn't make big change tbh
        self.k = k
        self.concat_input = concat_input

        self.coord_mlp = nn.Linear(in_features, self.s_dim)
        self.feature_mlp = nn.Linear(in_features, hidden_features)
        update_in = in_features + hidden_features if concat_input else hidden_features
        self.update_mlp = nn.Linear(update_in, hidden_features)
        self.residual_lin = nn.Linear(in_features, hidden_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = x.device
        B, H, F = x.shape
        out_padded = torch.zeros(B, H, self.hidden_features, device=device, dtype=x.dtype)

        # --- Flatten batch ---
        flat_mask = mask.flatten()
        valid_idx = torch.nonzero(flat_mask).squeeze(1)
        x_flat = x.reshape(-1, F)[valid_idx]
        s_flat = self.coord_mlp(x_flat)
        N = s_flat.shape[0]

        if N == 0:
            return out_padded
        if N == 1:
            out_b = self.update_mlp(self.feature_mlp(x_flat))
            out_padded.reshape(-1, self.hidden_features)[valid_idx] = out_b
            return out_padded

        # --- Fast path for s_dim == 1 ---
        if self.s_dim == 1:
            sort_vals = s_flat.squeeze(1)  # [N]
            sorted_idx = torch.argsort(sort_vals)
            x_sorted = x_flat[sorted_idx]

            half_k = self.k // 2
            # build symmetric offsets around each index
            neighbor_offsets = torch.arange(-half_k, half_k + (self.k % 2), device=device)
            neighbor_offsets = neighbor_offsets[neighbor_offsets != 0]  # exclude self

            arangeN = torch.arange(N, device=device).unsqueeze(1)
            neighbors_sorted = (arangeN + neighbor_offsets.unsqueeze(0)).clamp(0, N - 1)
            k_eff = min(self.k, neighbors_sorted.shape[1])
            final_neighbors = neighbors_sorted[:, :k_eff]
        else:
            # --- Pick random axis for presorting ---
            axis = torch.randint(0, self.s_dim, (1,), dtype=torch.long, device=device)
            sort_vals = s_flat.index_select(1, axis).squeeze(1)
            sorted_idx = torch.argsort(sort_vals)
            s_sorted = s_flat[sorted_idx]
            x_sorted = x_flat[sorted_idx]

            # --- Candidate neighbors: 2*k in sorted 1D axis ---
            neighbor_offsets = torch.arange(-self.k, self.k + 1, device=device)
            neighbor_offsets = neighbor_offsets[neighbor_offsets != 0]
            arangeN = torch.arange(N, device=device).unsqueeze(1)
            neighbors_sorted = (arangeN + neighbor_offsets.unsqueeze(0)).clamp(0, N - 1)

            # --- Full distance-based kNN in s_dim space ---
            s_src = s_sorted.unsqueeze(1)           # [N,1,s_dim]
            s_dst = s_sorted[neighbors_sorted]      # [N,2*k,s_dim]
            dist_sq = ((s_src - s_dst) ** 2).sum(dim=2)

            k_eff = min(self.k, dist_sq.shape[1])
            _, topk_idx = torch.topk(-dist_sq, k=k_eff, dim=1)
            final_neighbors = neighbors_sorted.gather(1, topk_idx)

        # --- Flatten for message passing ---
        src_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_eff).reshape(-1)
        dst_idx = final_neighbors.reshape(-1)

        # --- Message passing ---
        feat_sorted = self.feature_mlp(x_sorted)
        messages = feat_sorted[dst_idx]
        agg = torch.zeros(N, self.hidden_features, device=device, dtype=messages.dtype)
        counts = torch.zeros(N, device=device, dtype=messages.dtype)
        agg.index_add_(0, src_idx, messages)
        counts.index_add_(0, src_idx, torch.ones_like(src_idx, dtype=messages.dtype))
        counts = counts + (counts == 0).float()
        agg = agg / counts.unsqueeze(1)

        # --- Update step ---
        update_input = torch.cat([x_sorted, agg], dim=1) if self.concat_input else agg
        out_sorted = self.update_mlp(update_input) + self.residual_lin(x_sorted)

        # --- Scatter back to original order ---
        inv = torch.empty(N, dtype=torch.long, device=device)
        inv[sorted_idx] = torch.arange(N, device=device)
        out_flat = out_sorted[inv]

        out_padded.reshape(-1, self.hidden_features)[valid_idx] = out_flat
        return out_padded

class ScriptableClassifier(nn.Module):
    def __init__(self, in_features, hidden_features=64, num_layers=2, k=16):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.k = k

        # GravNet layers with skip connections (first expects in_features -> hidden_features)
        self.convs = nn.ModuleList()
        self.convs.append(EfficientGravNetLayer(in_features, hidden_features, k=k, concat_input=True))
        for _ in range(num_layers - 1):
            # subsequent layers take hidden_features as input
            self.convs.append(EfficientGravNetLayer(hidden_features, hidden_features, k=k, concat_input=False))

        # Final MLP with skip connections: concatenate original input + all layer outputs
        final_dim = in_features + num_layers * hidden_features
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 1)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: [batch, max_hits, in_features]
        mask: [batch, max_hits]
        returns: [batch, max_hits] probabilities
        """
        xs = [x]
        for conv in self.convs:
            x = conv(x, mask)
            xs.append(x)
        x = torch.cat(xs, dim=2)  # [batch, max_hits, final_dim]

        logits = self.mlp(x).squeeze(-1)  # [batch, max_hits]
        # Zero out padded positions
        logits = logits * mask
        return torch.sigmoid(logits)


# -----------------------------
# Lightning wrapper
# -----------------------------
class Classifier(pl.LightningModule):
    def __init__(self, in_features, hidden_features=64, num_layers=2, lr=1e-3, k=16):
        super().__init__()
        self.save_hyperparameters()
        self.model = ScriptableClassifier(in_features, hidden_features, num_layers, k)
        self.lr = lr
        self.in_features = in_features

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, y, weights = batch  # weights contain the class balancing
        mask = (weights > 0).float()  # Non-zero weights indicate valid hits
        probs = self(x, mask)
        
        # Apply mask to only compute loss on valid hits
        valid_mask = mask.bool()
        valid_probs = probs[valid_mask]
        valid_y = y[valid_mask].float()
        valid_weights = weights[valid_mask]
        
        # Weighted BCE loss for class imbalance
        loss = F.binary_cross_entropy(valid_probs, valid_y, weight=valid_weights)
        
        preds = (valid_probs >= 0.5).long()
        acc = (preds == y[valid_mask]).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        mask = (weights > 0).float()  # Non-zero weights indicate valid hits
        probs = self(x, mask)
        
        # Apply mask to only compute loss on valid hits
        valid_mask = mask.bool()
        valid_probs = probs[valid_mask]
        valid_y = y[valid_mask].float()
        valid_weights = weights[valid_mask]
        
        loss = F.binary_cross_entropy(valid_probs, valid_y, weight=valid_weights)
        
        preds = (valid_probs >= 0.5).long()
        acc = (preds == y[valid_mask]).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # -----------------------------
    # Export / Load
    # -----------------------------
    def export_to_torchscript(self, path: str):
        self.model.to("cpu")
        self.model.eval()
        #example_x = torch.randn(1, 450, self.in_features)
        #example_mask = torch.ones(1, 450)
        #traced = torch.jit.trace(self.model, (example_x, example_mask))
        #torch.jit.save(traced, path)
        torchscript_model = torch.jit.script(self.model)
        torchscript_model.save(path)
        print(f"TorchScript model saved to {path}")

    @staticmethod
    def load_from_torchscript(path: str, in_features: int):
        core_model = torch.jit.load(path, map_location="cpu")
        # Create wrapper with correct hyperparameters
        model = Classifier(in_features=in_features, hidden_features=64, num_layers=4, k=30)
        model.model = core_model
        model.eval()
        return model


# -----------------------------
# Loss Tracker Callback
# -----------------------------
class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())