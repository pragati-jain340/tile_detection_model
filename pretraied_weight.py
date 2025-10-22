import torch

ckpt_path = r"C:\pragati\codes\lightglue\SuperPoint\weights\superpoint_v6_from_tf.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n✅ TOTAL LAYERS:", len(ckpt))
print("🔑 EXAMPLE KEYS:")
for i, k in enumerate(ckpt.keys()):
    print(f"{i:02d}: {k}")
    if i > 30: break  # print first 30 only
