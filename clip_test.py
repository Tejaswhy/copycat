# -----------------------------
# 🔧 FIX SSL ISSUE
# -----------------------------
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# 📦 IMPORTS
# -----------------------------
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# 🔹 DEVICE
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------
# 🔹 LOAD CLIP
# -----------------------------
print("🚀 Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# 🔹 LABELS
# -----------------------------
labels = [
    "a cricket match with bat and wicket on pitch",
    "a football soccer match on grass field with goalpost",
    "a volleyball match with net and indoor court",
    "a formula 1 race car on racing track",
    "a baseball game with bat and bases",
    "a basketball game with hoop and court",
    "a hockey match with stick and goal",
    "a kabaddi match with players tackling on mat",
    "a rugby match with players tackling oval ball",
    "a tennis match with racket and net"
]

text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

# -----------------------------
# 🔹 VIDEO PATH
# -----------------------------
video_path = "/Users/tejasy/Documents/GDG Atria/output_all/f1/clip_2.mp4"

# -----------------------------
# 🔹 READ VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"❌ Cannot open video: {video_path}")

frame_embeddings = []
frame_count = 0

# 🔥 SSIM storage
prev_gray = None
ssim_scores = []

print("📹 Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Compute SSIM
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        score = ssim(prev_gray, gray)
        ssim_scores.append(score)

    prev_gray = gray

    # 🔹 CLIP sampling
    if frame_count % 10 == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)

        frame_embeddings.append(image_features.cpu().numpy())

    frame_count += 1

cap.release()

print("Frames extracted:", len(frame_embeddings))

# -----------------------------
# 🔹 SSIM RESULT
# -----------------------------
if len(ssim_scores) > 0:
    avg_ssim = np.mean(ssim_scores)
else:
    avg_ssim = 0

print(f"📊 Average SSIM: {avg_ssim:.4f}")

# -----------------------------
# 🔹 SAFETY CHECK
# -----------------------------
if len(frame_embeddings) == 0:
    raise ValueError("❌ No frames extracted — video might be corrupted")

# -----------------------------
# 🔹 VIDEO EMBEDDING
# -----------------------------
video_embedding = np.mean(frame_embeddings, axis=0)
video_embedding = video_embedding / np.linalg.norm(video_embedding)

# -----------------------------
# 🔹 TEXT FEATURES
# -----------------------------
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

text_features = text_features.cpu().numpy()
text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

# -----------------------------
# 🔹 SIMILARITY
# -----------------------------
similarity = 100 * (video_embedding @ text_features.T)

# -----------------------------
# 🔹 SOFTMAX
# -----------------------------
exp_sim = np.exp(similarity - np.max(similarity))
probs = exp_sim / np.sum(exp_sim)

# -----------------------------
# 🔹 RESULT
# -----------------------------
best_idx = np.argmax(probs)
best_label = labels[best_idx]

print("\n🎯 Predicted Sport:", best_label)

print("\n📊 Probabilities:")
for i, label in enumerate(labels):
    print(f"{label:40s}: {probs[0][i]*100:.2f}%")

print("\n🎉 DONE")