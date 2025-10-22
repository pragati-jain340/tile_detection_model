import cv2, numpy as np, json, os

# ================================
# 1️⃣ Load Original Image & Keypoints
# ================================
img_path = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train\Img46.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Define keypoints (tile corners)
keypoints = np.array([
    [0,0],[89,0],[178,0],[267,0],[356,0],[445,0],
    [0,89],[89,89],[178,89],[267,89],[356,89],[445,89],
    [0,178],[89,178],[178,178],[267,178],[356,178],[445,178],
    [0,267],[89,267],[178,267],[267,267],[356,267],[445,267],
    [0,356],[89,356],[178,356],[267,356],[356,356],[445,356],
    [0,445],[89,445],[178,445],[267,445],[356,445],[445,445]
], np.float32)


# ================================
# 2️⃣ Function to apply random warp
# ================================
def safe_random_warp(img, keypoints, max_shift=25, pad=60):
    h, w = img.shape[:2]

    # 1️⃣ Pad the image (reflection avoids black borders)
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h2, w2 = img_pad.shape[:2]

    # 2️⃣ Define corners before warp
    pts1 = np.float32([[pad, pad], [w+pad, pad], [pad, h+pad], [w+pad, h+pad]])

    # 3️⃣ Generate random outward warp
    shift = np.random.uniform(-max_shift, max_shift, (4,2)).astype(np.float32)
    pts2 = pts1 + shift

    # 4️⃣ Compute homography and warp image
    H = cv2.getPerspectiveTransform(pts1, pts2)
    warped_full = cv2.warpPerspective(img_pad, H, (w2, h2))

    # 5️⃣ Find bounding box of transformed corners (to crop safely)
    warped_corners = cv2.perspectiveTransform(pts1[None,:,:], H)[0]
    x_min, y_min = warped_corners.min(axis=0).astype(int)
    x_max, y_max = warped_corners.max(axis=0).astype(int)

    # 6️⃣ Clamp bounds inside image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w2, x_max)
    y_max = min(h2, y_max)

    # 7️⃣ Crop adaptively to keep visible content
    warped_cropped = warped_full[y_min:y_max, x_min:x_max]
    warped_cropped = cv2.resize(warped_cropped, (w, h))  # resize back to 256×256

    # 8️⃣ Warp keypoints and shift to crop coords
    warped_kps = cv2.perspectiveTransform((keypoints + pad)[None,:,:], H)[0]
    warped_kps -= [x_min, y_min]
    warped_kps *= [w / (x_max - x_min), h / (y_max - y_min)]  # rescale

    # 9️⃣ Remove invalid points
    valid = (
        (warped_kps[:,0] >= 0) & (warped_kps[:,0] < w) &
        (warped_kps[:,1] >= 0) & (warped_kps[:,1] < h)
    )
    warped_kps = warped_kps[valid]

    return warped_cropped, warped_kps, H




# ================================
# 3️⃣ Apply warp
# ================================
warped_img, warped_kps, H = safe_random_warp(img, keypoints)

# ================================
# 4️⃣ Save warped image and JSON
# ================================
save_dir = r"C:\pragati\codes\lightglue\tile_detection_model\dataset_train"
os.makedirs(os.path.join(save_dir), exist_ok=True)

data = {
    "imageA": "Img46.jpg",
    "imageB": "Img46_warped.jpg",
    "keypointsA": keypoints.tolist(),
    "keypointsB": warped_kps.tolist(),
    "H": H.tolist()
}

json_path = os.path.join(save_dir, "Img46_pair.json")
imgB_path = os.path.join(save_dir, "Img46_warped.jpg")

with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

cv2.imwrite(imgB_path, warped_img)
print(f"✅ Saved warped image at: {imgB_path}")
print(f"✅ Saved JSON file at: {json_path}")
print(f"Image sizes → Original: {img.shape}, Warped: {warped_img.shape}")
