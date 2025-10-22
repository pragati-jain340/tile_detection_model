# model_loader.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SuperPoint architecture once, load the right weights once, and let any other script reuse that already-loaded model.

# ===============================
# SuperPoint Model Definition
# ===============================
class SuperPoint(nn.Module):
    def __init__(self, desc_dim=256):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        chans = [64, 64, 128, 128, 256, 256]
        self.convs = nn.ModuleList()
        in_ch = 1
        for c in chans:
            self.convs.append(nn.Conv2d(in_ch, c, 3, padding=1))
            in_ch = c
        self.det_head = nn.Conv2d(256, 1, 1)
        self.desc_head = nn.Conv2d(256, desc_dim, 1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.relu(conv(x))
            if i % 2 == 1:
                x = self.pool(x)
        shared = x
        det = torch.sigmoid(self.det_head(shared))
        desc = F.normalize(self.desc_head(shared), p=2, dim=1)
        return det, desc


# ===============================
# Model Loader Function
# ===============================
def load_superpoint_model(weights_path=None, device=None):
    """
    Loads the SuperPoint model (fine-tuned or pretrained) only once.

    Args:
        weights_path (str): Path to model weights.
        device (torch.device): CPU or GPU.

    Returns:
        model (nn.Module): Loaded SuperPoint model ready for inference/training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperPoint().to(device)

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded SuperPoint weights from: {weights_path}")
    else:
        print("⚠️ No weights found — loading untrained model.")

    model.eval()
    return model


# ===============================
# Singleton Loader (Preload Once)
# ===============================
_model_instance = None

def get_superpoint_model(weights_path=None):
    """Return a preloaded global model instance (singleton)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = load_superpoint_model(weights_path)
    return _model_instance


# ===============================
# Test Loader
# ===============================
if __name__ == "__main__":
    model = get_superpoint_model(
        weights_path=r"C:\pragati\codes\lightglue\tile_detection_model\checkpoints\superpoint_tiles_finetuned.pth"
    )
    print("✅ SuperPoint model loaded and ready for use.")
