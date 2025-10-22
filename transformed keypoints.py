import cv2
import numpy as np
import json
import os

# ================================
# 1️⃣ Load base dataset (real_images.json)
# ================================
real_json_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train\real_images.json"
save_dir = os.path.dirname(real_json_path)

# Folder paths
real_dir = os.path.join(save_dir, "real_marked_images")
warped_dir = os.path.join(save_dir, "warped_images")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(warped_dir, exist_ok=True)

with open(real_json_path, "r") as f:
    real_data = json.load(f)

images_info = real_data["images"]
print(f"✅ Loaded {len(images_info)} base images from {real_json_path}")

# ================================
# 2️⃣ Helper: Random perspective warp
# ================================
def safe_random_warp(img, keypoints, max_shift=10, strong_warp=False):
    """
    Apply a mild random perspective warp with consistent keypoints.
    Keeps image the same size (no cropping/padding distortion).
    """
    h, w = img.shape[:2]

    # 1️⃣ Define original corners
    pts1 = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    # 2️⃣ Generate random shifts (gentle)
    shift = np.random.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32)
    pts2 = pts1 + shift

    # 3️⃣ Compute homography
    H = cv2.getPerspectiveTransform(pts1, pts2)

    # 4️⃣ Warp image directly (no resize)
    border_type = cv2.BORDER_REPLICATE if strong_warp else cv2.BORDER_REFLECT
    warped_img = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_type
    )

    # 5️⃣ Warp keypoints
    warped_kps = cv2.perspectiveTransform(keypoints[None, :, :], H)[0]

    # 6️⃣ Keep valid points inside image
    warped_kps[:, 0] = np.clip(warped_kps[:, 0], 0, w - 1)
    warped_kps[:, 1] = np.clip(warped_kps[:, 1], 0, h - 1)

    return warped_img, warped_kps, H


# ================================
# 3️⃣ Create warps and JSON pairs
# ================================
pairs_data = []

for entry in images_info:
    img_id = entry["id"]
    img_name = entry["filename"]

    # Use real_marked_images folder
    img_path = os.path.join(real_dir, img_name)
    keypoints = np.array(entry["keypoints"], np.float32)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Missing image: {img_path}")
        continue

    warped_img, warped_kps, H = safe_random_warp(img, keypoints)

    warped_name = os.path.splitext(img_name)[0] + "_warped.jpg"
    warped_path = os.path.join(warped_dir, warped_name)
    cv2.imwrite(warped_path, warped_img)

    print(f"✅ Warped {img_name} → {warped_name}")

    pair_entry = {
        "id": f"pair_{img_id}",
        "source_id": img_id,
        "image1": os.path.join("real_marked_images", img_name).replace("\\", "/"),
        "image2": os.path.join("warped_images", warped_name).replace("\\", "/"),
        "keypoints2": warped_kps.tolist(),
        "H": H.tolist()
    }
    pairs_data.append(pair_entry)

# ================================
# 4️⃣ Save as train_pairs.json
# ================================
output_path = os.path.join(save_dir, "train_pairs.json")
with open(output_path, "w") as f:
    json.dump({"pairs": pairs_data}, f, indent=4)

print(f"\n💾 Saved {len(pairs_data)} warped pairs → {output_path}")
