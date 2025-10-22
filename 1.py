import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from Dataset_Loader import TilePairDataset
import os
from tqdm import tqdm

# ===============================
# 1️⃣ SuperPoint Architecture
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
# 2️⃣ Losses
# ===============================
def descriptor_loss(descA, descB, H):
    return ((descA - descB) ** 2).mean()

# ===============================
# 3️⃣ Dataset Split (Train / Validation)
# ===============================
dataset_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"
full_dataset = TilePairDataset(dataset_path)

# 80/20 split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"✅ Dataset split → {len(train_dataset)} train, {len(val_dataset)} val samples.")

# ===============================
# 4️⃣ Model + Pretrained Weights
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperPoint().to(device)

pretrained_path = r"C:\pragati\codes\lightglue\SuperPoint\weights\superpoint_v6_from_tf.pth"
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Loaded pretrained SuperPoint weights from {pretrained_path}")
else:
    print("⚠️ Pretrained weights not found — starting from scratch.")

# Freeze all encoder layers except last 2
for param in model.convs.parameters():
    param.requires_grad = False

# unfreezing last 2 conv layers
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
# 5️⃣ Evaluation Function
# ===============================
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgA, imgB, gtA, gtB, H in dataloader:
            imgA, imgB, gtA, gtB, H = imgA.to(device), imgB.to(device), gtA.to(device), gtB.to(device), H.to(device)
            detA, descA = model(imgA)
            detB, descB = model(imgB)
            detA_up = F.interpolate(detA, size=gtA.shape[-2:], mode="bilinear", align_corners=False)
            detB_up = F.interpolate(detB, size=gtB.shape[-2:], mode="bilinear", align_corners=False)
            det_loss = F.binary_cross_entropy(detA_up, gtA) + F.binary_cross_entropy(detB_up, gtB)
            desc_loss = descriptor_loss(descA, descB, H)
            total_loss += (det_loss + 1e-4 * desc_loss).item()
    return total_loss / len(dataloader)

# ===============================
# 6️⃣ Training Loop + Early Stopping
# ===============================
EPOCHS = 50
patience = 5
best_val_loss = float("inf")
no_improve = 0

save_dir = "tile_detection_model/checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for imgA, imgB, gtA, gtB, H in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        imgA, imgB, gtA, gtB, H = imgA.to(device), imgB.to(device), gtA.to(device), gtB.to(device), H.to(device)

        detA, descA = model(imgA)
        detB, descB = model(imgB)
        detA_up = F.interpolate(detA, size=gtA.shape[-2:], mode="bilinear", align_corners=False)
        detB_up = F.interpolate(detB, size=gtB.shape[-2:], mode="bilinear", align_corners=False)
        det_loss = F.binary_cross_entropy(detA_up, gtA) + F.binary_cross_entropy(detB_up, gtB)
        desc_loss = descriptor_loss(descA, descB, H)
        total_loss = det_loss + 1e-4 * desc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, device)

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] → train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, "best_superpoint.pth"))
        print("💾 Model improved → saved best checkpoint.")
    else:
        no_improve += 1
        print(f"⏳ No improvement for {no_improve}/{patience} epochs.")
        if no_improve >= patience:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n🏁 Training complete. Best model saved at {os.path.join(save_dir, 'best_superpoint.pth')}")
