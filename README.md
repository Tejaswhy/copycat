AI-Powered Sports Video Copyright Detection

This project is an AI-powered copyright detection system designed to identify visually similar sports videos by combining deep learning and traditional computer vision techniques. It helps determine whether an uploaded sports video closely matches existing copyrighted content stored in a reference database.

The application begins by allowing users to upload a video through an interactive Streamlit interface. A CLIP (Contrastive Language–Image Pre-training) model analyzes key video frames and classifies the uploaded content into sports such as Football, Cricket, Basketball, Baseball, Tennis, Hockey, Rugby, Kabaddi, Volleyball, and Formula 1. This semantic understanding provides context before similarity analysis begins.

To detect potential copyright infringement, the system extracts representative frames using OpenCV and compares them with videos stored in the database. Frame similarity is measured using Perceptual Hashing (pHash) to identify visually similar content despite minor modifications, while Structural Similarity Index (SSIM) evaluates structural and visual consistency between frames. The combined similarity score is used to identify the closest matching video.

For authenticity verification, the system also supports Rivagan digital watermarking, enabling invisible watermark embedding and extraction for content ownership verification. The application presents the uploaded video alongside the best-matching database video, displays similarity metrics, and classifies the result as either Clean, Possible Copyright, or High-Risk Copyright.

Technologies Used:
Python
Streamlit
PyTorch
OpenAI CLIP (ViT-B/32)
OpenCV
Perceptual Hashing (pHash)
Structural Similarity Index (SSIM)
Rivagan Watermarking
NumPy
Pandas

Pillow

This project demonstrates the integration of Artificial Intelligence, Deep Learning, Computer Vision, and Digital Watermarking to build a scalable and practical solution for sports video copyright detection and similarity analysis.
