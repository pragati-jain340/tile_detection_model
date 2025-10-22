import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from Dataset_Loader import TilePairDataset
from model_loader import get_superpoint_model
import os
from tqdm import tqdm
import numpy as np
import cv2


# ===============================
# 1️⃣ Dataset Split (Train / Validation)
# ===============================
dataset_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"
full_dataset = TilePairDataset(
    dataset_dir=dataset_path,
    use_rgb=False,          # Keep grayscale
    include_heatmaps=False   # Include keypoint heatmaps
)

# 80/20 train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"✅ Dataset split → {len(train_dataset)} train, {len(val_dataset)} val samples.")


# ===============================
# 2️⃣ Load Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_path = r"C:\pragati\codes\lightglue\SuperPoint\weights\superpoint_v6_from_tf.pth"
model = get_superpoint_model(pretrained_path).to(device)
print(f"✅ Loaded SuperPoint model on {device}")


# ===============================
# 3️⃣ Freeze + Unfreeze Layers
# ===============================
for param in model.convs.parameters():
    param.requires_grad = False  # freeze encoder

# Unfreeze last 2 convs + heads
for param in list(model.convs[-2].parameters()) + list(model.convs[-1].parameters()):
    param.requires_grad = True

trainable_params = (
    list(model.convs[-2].parameters()) +
    list(model.convs[-1].parameters()) +
    list(model.det_head.parameters()) +
    list(model.desc_head.parameters())
)
optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
print(f"🔧 Training {sum(p.numel() for p in trainable_params)} parameters (heads + last convs)")


# ===============================
# 4️⃣ Loss Functions
# ===============================
def detector_loss(det, gt):
    """Binary cross-entropy for keypoint heatmaps."""
    return F.binary_cross_entropy(det, gt)


def descriptor_loss(descA, descB, H):
    """
    Descriptor loss using homography to align B’s descriptors to A.
    Ensures spatial consistency under geometric warp.
    """
    B, C, Hh, Ww = descA.shape
    total_loss = 0.0

    for b in range(B):
        H_np = H[b].detach().cpu().numpy()
        if H_np.shape == (1, 3, 3):
            H_np = H_np[0]
        elif H_np.shape != (3, 3):
            raise ValueError(f"Invalid homography shape: {H_np.shape}")
        
        
        grid = np.stack(np.meshgrid(np.arange(Ww), np.arange(Hh)), axis=-1).astype(np.float32)
        # Flatten grid (HxW, 2)
        grid_flat = grid.reshape(-1, 2)[None, :, :]  # (1, H*W, 2)
        
        # Apply homography
        grid_warp_flat = cv2.perspectiveTransform(grid_flat, H_np)[0]  # (H*W, 2)
        
        # Reshape back to (H, W, 2)
        grid_warp = grid_warp_flat.reshape(Hh, Ww, 2)

        # Normalize for grid_sample
        grid_warp[..., 0] = (grid_warp[..., 0] / (Ww - 1)) * 2 - 1
        grid_warp[..., 1] = (grid_warp[..., 1] / (Hh - 1)) * 2 - 1
        grid_torch = torch.from_numpy(grid_warp).to(descB.device)[None, ...]

        warped_descB = F.grid_sample(descB[b:b+1], grid_torch, align_corners=False)
        total_loss += F.mse_loss(descA[b:b+1], warped_descB)

    return total_loss / B


# ===============================
# 5️⃣ Evaluation Function
# ===============================
def evaluate(model, dataloader, device):
    model.eval()
    total_det_loss, total_desc_loss = 0.0, 0.0

    with torch.no_grad():
        for imgA, imgB, gtA, gtB, H in dataloader:
            imgA, imgB, gtA, gtB, H = imgA.to(device), imgB.to(device), gtA.to(device), gtB.to(device), H.to(device)

            detA, descA = model(imgA)
            detB, descB = model(imgB)

            detA_up = F.interpolate(detA, size=gtA.shape[-2:], mode="bilinear", align_corners=False)
            detB_up = F.interpolate(detB, size=gtB.shape[-2:], mode="bilinear", align_corners=False)

            det_loss = detector_loss(detA_up, gtA) + detector_loss(detB_up, gtB)
            desc_loss = descriptor_loss(descA, descB, H)

            total_det_loss += det_loss.item()
            total_desc_loss += desc_loss.item()

    avg_det = total_det_loss / len(dataloader)
    avg_desc = total_desc_loss / len(dataloader)
    return avg_det, avg_desc


# ===============================
# 6️⃣ Training Loop + Early Stopping
# ===============================
EPOCHS = 50
patience = 5
best_val_loss = float("inf")
no_improve = 0

save_dir = r"C:\pragati\codes\lightglue\tile_detection_model\checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_det_loss, total_desc_loss = 0.0, 0.0

    for imgA, imgB, gtA, gtB, H in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        imgA, imgB, gtA, gtB, H = imgA.to(device), imgB.to(device), gtA.to(device), gtB.to(device), H.to(device)

        detA, descA = model(imgA)
        detB, descB = model(imgB)

        detA_up = F.interpolate(detA, size=gtA.shape[-2:], mode="bilinear", align_corners=False)
        detB_up = F.interpolate(detB, size=gtB.shape[-2:], mode="bilinear", align_corners=False)

        det_loss = detector_loss(detA_up, gtA) + detector_loss(detB_up, gtB)
        desc_loss = descriptor_loss(descA, descB, H)
        total_loss = det_loss + 1e-4 * desc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_det_loss += det_loss.item()
        total_desc_loss += desc_loss.item()

    avg_train_det = total_det_loss / len(train_loader)
    avg_train_desc = total_desc_loss / len(train_loader)
    val_det, val_desc = evaluate(model, val_loader, device)
    val_loss = val_det + 1e-4 * val_desc

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] → "
          f"train_det={avg_train_det:.4f}, val_det={val_det:.4f} | "
          f"train_desc={avg_train_desc:.4f}, val_desc={val_desc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, "superpoint_tiles_finetuned.pth"))
        print("💾 Model improved → saved best checkpoint.")
    else:
        no_improve += 1
        print(f"⏳ No improvement for {no_improve}/{patience} epochs.")
        if no_improve >= patience:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n🏁 Training complete. Best model saved at {os.path.join(save_dir, 'superpoint_tiles_finetuned.pth')}")
