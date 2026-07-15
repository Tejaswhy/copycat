import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
import imagehash

from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import clip

# =====================================================
# CONFIG
# =====================================================

BASE_PATH = "/Users/tejasy/Documents/GDG Atria/DATABASE"

st.set_page_config(
    page_title="AI Copyright Detector",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<h1 style='text-align:center;color:#2E86C1;'>
🎬 AI Video Copyright Detection
</h1>

<p style='text-align:center;font-size:18px'>
Upload a sports video and compare it against your copyright database.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Video"

)

# =====================================================
# CLIP
# =====================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

clip_model, preprocess = load_clip()

labels = [
    "cricket",
    "football",
    "basketball",
    "baseball",
    "f1",
    "tennis",
    "rugby",
    "kabaddi",
    "hockey",
    "volleyball"
]

text_inputs = torch.cat(
    [clip.tokenize(f"a {x} match") for x in labels]
).to(device)

# =====================================================
# SAVE VIDEO
# =====================================================

def save_temp(file):

    path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    with open(path,"wb") as f:
        f.write(file.read())

    return path

# =====================================================
# CLIP PREDICTION
# =====================================================
def predict_label(video):

    cap = cv2.VideoCapture(video)

    frames = []

    # Read first 5 frames
    for _ in range(5):

        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    cap.release()

    # If no frames could be read
    if len(frames) == 0:
        return "Unknown", 0.0

    features = []

    # Extract CLIP features
    for frame in frames:

        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        inp = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(inp)

        features.append(feat.cpu().numpy())

    # Average all frame embeddings
    video_feature = np.mean(features, axis=0)

    # Normalize safely
    norm = np.linalg.norm(video_feature)

    if norm > 0:
        video_feature = video_feature / norm

    # Encode text labels
    with torch.no_grad():
        text_feature = clip_model.encode_text(text_inputs)

    text_feature = text_feature.cpu().numpy()

    # Normalize text embeddings safely
    text_norm = np.linalg.norm(
        text_feature,
        axis=1,
        keepdims=True
    )

    text_feature = text_feature / np.maximum(text_norm, 1e-8)

    # Compute cosine similarity
    similarity = video_feature @ text_feature.T

    similarity = similarity[0]

    # Best matching label
    idx = np.argmax(similarity)

    confidence = float(similarity[idx])

    return labels[idx], confidence

# =====================================================
# FRAME EXTRACTION
# =====================================================

def get_frames(video,n=3):

    cap=cv2.VideoCapture(video)

    frames=[]

    for _ in range(n):

        ret,frame=cap.read()

        if ret:

            frames.append(frame)

    cap.release()

    return frames

# =====================================================
# FEATURES
# =====================================================

def get_phash(frame):

    image=Image.fromarray(
        cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    )

    return imagehash.phash(image)

def compute_ssim(frame1,frame2):

    frame2=cv2.resize(
        frame2,
        (frame1.shape[1],frame1.shape[0])
    )

    return ssim(
        cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    )

# =====================================================
# MATCH DATABASE
# =====================================================

def match_video(video,progress):

    input_frames=get_frames(video)

    results=[]

    videos=[]

    for root,_,files in os.walk(BASE_PATH):

        for file in files:

            if file.lower().endswith(
                (".mp4",".avi",".mov")
            ):

                videos.append(
                    os.path.join(root,file)
                )

    total=len(videos)

    for index,path in enumerate(videos):

        progress.progress((index+1)/total)

        db_frames=get_frames(path)

        if len(db_frames)==0:

            continue

        phash=min(

            get_phash(f1)-get_phash(f2)

            for f1,f2 in zip(
                input_frames,
                db_frames
            )

        )

        ssim_value=np.mean([

            compute_ssim(f1,f2)

            for f1,f2 in zip(
                input_frames,
                db_frames
            )

        ])

        phash_score=max(
            0,
            1-phash/32
        )

        ssim_score=max(
            0,
            min(ssim_value,1)
        )

        score=(

            phash_score*0.5

            +

            ssim_score*0.5

        )

        results.append({

            "Video":os.path.basename(path),

            "Path":path,

            "Score":round(score,3),

            "pHash":round(phash_score,3),

            "SSIM":round(ssim_score,3)

        })

    results=sorted(

        results,

        key=lambda x:x["Score"],

        reverse=True

    )

    return results

# =====================================================
# MAIN UI
# =====================================================

if uploaded_file:

    if "uploaded_path" not in st.session_state:
     st.session_state.uploaded_path = save_temp(uploaded_file)

    path = st.session_state.uploaded_path

    # -------------------------------
    # Uploaded Video Preview
    # -------------------------------

    st.subheader("📤 Uploaded Video")

    st.video(path)

    st.divider()

    # -------------------------------
    # AI Prediction
    # -------------------------------

    with st.spinner("🤖 Running CLIP Prediction..."):

        label, confidence = predict_label(path)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Detected Sport",
            label.title()
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    st.info(
        f"AI predicts that this video is **{label.title()}**."
    )

    st.divider()

    # -------------------------------
    # Analyze Button
    # -------------------------------

    if st.button(
        "🚀 Analyze Copyright",
        use_container_width=True
    ):

        with st.spinner("Searching video database..."):

            progress = st.progress(0)

            results = match_video(path, progress)

            progress.empty()

        if len(results) == 0:

            st.error("No videos found in database.")

            st.stop()

        best = results[0]
        st.json(best)
        st.write("Matched Video Path:")
        st.code(best["Path"])

        st.write("File Exists:", os.path.exists(best["Path"]))

        st.divider()

        # ==========================================
        # COPYRIGHT RESULT
        # ==========================================

        if best["Score"] >= 0.90:

            st.error("🚨 HIGH RISK COPYRIGHT DETECTED")

        elif best["Score"] >= 0.75:

            st.warning("⚠️ POSSIBLE COPYRIGHT DETECTED")

        else:

            st.success("✅ VIDEO APPEARS CLEAN")

        # ==========================================
        # METRICS
        # ==========================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Overall",
                f"{best['Score']*100:.1f}%"
            )

        with c2:
            st.metric(
                "pHash",
                f"{best['pHash']*100:.1f}%"
            )

        with c3:
            st.metric(
                "SSIM",
                f"{best['SSIM']*100:.1f}%"
            )

        st.divider()

        # ==========================================
        # VIDEO COMPARISON
        # ==========================================

        st.subheader("🎥 Video Comparison")

        left, right = st.columns(2)

        with left:

            st.markdown("### 📤 Uploaded Video")

            with open(path, "rb") as f:
              uploaded_bytes = f.read()

            st.video(uploaded_bytes)

            st.caption(os.path.basename(path))

        with right:

            st.markdown("### 🎬 Best Match")

            with open(best["Path"], "rb") as f:
               matched_bytes = f.read()
            st.video(matched_bytes)
            st.write(os.path.exists(best["Path"]))

            st.caption(best["Video"])

        st.divider()

        # ==========================================
        # BEST MATCH INFO
        # ==========================================

        st.subheader("🏆 Best Match")

        st.write(f"**Video:** `{best['Video']}`")

        st.write(
            f"**Similarity:** {best['Score']*100:.2f}%"
        )

        st.progress(float(best["Score"]))

        st.write("")

        st.write("### pHash Similarity")

        st.progress(float(best["pHash"]))

        st.write("")

        st.write("### SSIM Similarity")

        st.progress(float(best["SSIM"]))

        st.divider()

        # ==========================================
        # TOP MATCHES
        # ==========================================

        st.subheader("📋 Top 5 Similar Videos")

        df = pd.DataFrame(results[:5])

        df.index = np.arange(
            1,
            len(df) + 1
        )

        st.dataframe(
            df[
                [
                    "Video",
                    "Score",
                    "pHash",
                    "SSIM"
                ]
            ],
            use_container_width=True
        )