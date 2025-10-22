import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset_Loader import TilePairDataset
from model_loader import SuperPoint  #  import model from loader
import os
from tqdm import tqdm

# ===============================
# 1️⃣ Dataset Setup
# ===============================
dataset_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"
dataset = TilePairDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print(f"✅ Found {len(dataset)} pairs → {len(dataset)*2} total images for fine-tuning.")


# ===============================
# 2️⃣ Load Model (and Pretrained Weights)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperPoint().to(device)

pretrained_path = r"C:\pragati\codes\lightglue\SuperPoint\weights\superpoint_v6_from_tf.pth"
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Loaded pretrained SuperPoint weights from {pretrained_path}")
else:
    print("⚠️ Pretrained weights not found — training from scratch.")


# ===============================
# 3️⃣ Freeze Encoder, Train Heads
# ===============================
for param in model.convs.parameters():
    param.requires_grad = False  # freeze backbone

# Unfreeze last two conv layers (Conv5, Conv6)
for param in list(model.convs[-2].parameters()) + list(model.convs[-1].parameters()):
    param.requires_grad = True

# Define optimizer for trainable params
trainable_params = (
    list(model.convs[-2].parameters())
    + list(model.convs[-1].parameters())
    + list(model.det_head.parameters())
    + list(model.desc_head.parameters())
)
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
print(f"🔒 Encoder frozen | 🔧 Training {sum(p.numel() for p in trainable_params)} head parameters.")


# ===============================
# 4️⃣ Loss Function
# ===============================
def descriptor_loss(descA, descB, H):
    return ((descA - descB) ** 2).mean()


# ===============================
# 5️⃣ Training Loop
# ===============================
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_det_loss, total_desc_loss = 0.0, 0.0

    print(f"\n🌀 Epoch {epoch+1}/{EPOCHS}")
    for i, (imgA, imgB, gtA, gtB, H) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")):
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

        total_det_loss += det_loss.item()
        total_desc_loss += desc_loss.item()

    avg_det = total_det_loss / len(dataloader)
    avg_desc = total_desc_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] → det_loss={avg_det:.4f}, desc_loss={avg_desc:.4f}")


# ===============================
# 6️⃣ Save Fine-Tuned Model
# ===============================
os.makedirs("tile_detection_model/checkpoints", exist_ok=True)
save_path = "tile_detection_model/checkpoints/superpoint_tiles_finetuned.pth"
torch.save(model.state_dict(), save_path)
print(f"\n💾 Fine-tuned model saved at {save_path}")
