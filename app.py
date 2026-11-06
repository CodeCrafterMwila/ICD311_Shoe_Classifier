import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models

import time
import pandas as pd
from datetime import datetime
from torchvision import transforms
from PIL import Image
from pathlib import Path
from io import BytesIO


# ==========================================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================================

st.set_page_config(
    page_title="CodeCrafterMwila | Shoe Classifier",
    page_icon="👟",
    layout="wide"
)

# Initialize session state for persistent sample selection
if "selected_sample_file" not in st.session_state:
    st.session_state.selected_sample_file = None

# Custom CSS for Dark GitHub Theme & Neon Green Accents
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    div[data-testid="stMetricValue"] {
        color: #00FF66;
    }
    </style>
""", unsafe_allow_html=True)

st.image(
    "assets/banner.png",
    use_column_width=True
)

st.markdown("""
Identify footwear categories using a custom-trained
**MobileNetV3 deep learning model** powered by
**PyTorch** and deployed with **Streamlit**.

Built with modern computer vision and deep learning
techniques to perform intelligent footwear recognition.
""")

st.divider()

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# IMAGE TRANSFORMATIONS
# ==========================================================

IMAGE_SIZE = 224

evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# MODEL PATH
# ==========================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "shoe_classifier_mobilenetv3.pth"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, class_names


try:
    with st.spinner("Loading MobileNetV3 model..."):
        model, class_names = load_model()

except Exception as error:
    st.error("Unable to load AI model.")
    st.exception(error)
    st.stop()

# ==========================================================
# DEVICE INFO
# ==========================================================

if device.type == "cuda":
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = "CPU"
    

# ==========================================================
# SIDEBAR & MLOPS CONTROLS
# ==========================================================

with st.sidebar:

    st.title("👟 Shoe Classifier")
    st.success("Model Loaded")

    st.write("### System")
    st.write(f"**Device:** {device_name}")
    st.write(f"**Classes:** {len(class_names)}")
    st.write(f"**Input Size:** {IMAGE_SIZE}px")

    st.divider()

    st.write("### ⚙️ Inference Settings")
    confidence_threshold = st.slider(
        "Confidence Cutoff (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        help="Predictions below this score will trigger an uncertainty flag."
    )

    st.divider()

    st.write("### Developer")
    st.write("🚀 CodeCrafterMwila")
    st.write("Deep Learning & Computer Vision")

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
Upload a shoe image (or multiple images) and let a trained **MobileNetV3**
deep learning model identify the most likely footwear category.

### Technologies
- PyTorch
- MobileNetV3
- Deep Learning
- Computer Vision
- Streamlit
""")

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.divider()
st.subheader("🖥️ AI System Dashboard")

dashboard1, dashboard2, dashboard3 = st.columns(3)

with dashboard1:
    st.metric("Compute Device", "GPU" if device.type == "cuda" else "CPU")

with dashboard2:
    st.metric("Model", "MobileNetV3")

with dashboard3:
    st.metric("Categories", len(class_names))

# ==========================================================
# UPLOAD SECTION & SAMPLE SELECTOR & LIVE CAMERA
# ==========================================================

st.divider()
st.subheader("📸 Input Selection")

mode_tab1, mode_tab2, mode_tab3 = st.tabs([
    "📤 Upload Custom Images", 
    "⚡ Instant Sample Gallery", 
    "📷 Live Camera Capture"
])

uploaded_files = []

with mode_tab1:
    files = st.file_uploader(
        "Choose shoe image(s) for single or batch processing",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if files:
        st.session_state.selected_sample_file = None
        uploaded_files = files

with mode_tab2:
    st.markdown("Click on any sample test image to run instant inference without uploading files.")
    
    sample_dir = Path(__file__).resolve().parent / "assets" / "samples"
    
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if sample_images:
            cols = st.columns(min(len(sample_images), 4))
            for idx, img_path in enumerate(sample_images[:4]):
                with cols[idx]:
                    st.image(str(img_path), caption=img_path.stem.replace("_", " ").title(), use_column_width=True)
                    if st.button(f"Test Sample #{idx+1}", key=f"sample_{idx}", use_container_width=True):
                        with open(img_path, "rb") as f:
                            sample_bytes = f.read()
                        sample_buffer = BytesIO(sample_bytes)
                        sample_buffer.name = img_path.name
                        st.session_state.selected_sample_file = sample_buffer
        else:
            st.caption("No sample images found in assets/samples directory.")
    else:
        st.caption("Add test images to assets/samples/ to unlock instant gallery mode.")

# Bind persistent sample selection if active
if st.session_state.selected_sample_file is not None and not uploaded_files:
    uploaded_files = [st.session_state.selected_sample_file]

with mode_tab3:
    st.markdown("Use your webcam or mobile camera to snap a live photo of a shoe.")
    camera_image = st.camera_input("Take a photo of your shoe")
    if camera_image is not None:
        st.session_state.selected_sample_file = None
        camera_image.name = "live_camera_snap.jpg"
        uploaded_files = [camera_image]

st.caption("Supported formats: JPG, JPEG, PNG")

# ==========================================================
# MAIN PREDICTION FLOW (SINGLE & BATCH)
# ==========================================================

if uploaded_files:

    # ------------------------------------------------------
    # SINGLE IMAGE MODE
    # ------------------------------------------------------
    if len(uploaded_files) == 1:
        single_file = uploaded_files[0]

        try:
            image = Image.open(
                BytesIO(single_file.getvalue())
            ).convert("RGB")
        except Exception:
            st.error("Could not load image.")
            st.stop()

        preview_col, info_col = st.columns([2, 1])

        with preview_col:
            st.image(
                image,
                caption=f"Target Image: {single_file.name}",
                use_column_width=True
            )

        with info_col:
            st.info("✅ Image loaded\n\nReady for classification")

        classify = st.button(
            "🚀 Run AI Classification",
            use_container_width=True,
            type="primary"
        )

        if classify:
            with st.spinner("Analysing image..."):
                start_time = time.perf_counter()

                image_tensor = (
                    evaluation_transform(image)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.inference_mode():
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    top_probs, top_indices = torch.topk(
                        probs,
                        k=min(3, len(class_names)),
                        dim=1
                    )

                latency = (time.perf_counter() - start_time) * 1000

            predicted_idx = top_indices[0][0].item()
            confidence = top_probs[0][0].item() * 100
            predicted_class = class_names[predicted_idx].replace("_", " ").title()

            st.divider()
            st.subheader("🎯 Classification Results")

            result1, result2, result3 = st.columns(3)

            with result1:
                st.metric("Predicted Class", predicted_class)

            with result2:
                st.metric("Confidence", f"{confidence:.2f}%")

            with result3:
                st.metric("Inference Latency", f"{latency:.2f} ms")

            # Dynamic threshold comparison
            if confidence >= confidence_threshold:
                st.success(f"✅ High confidence prediction (Exceeds {confidence_threshold}% cutoff)")
            elif confidence >= 50:
                st.warning(f"⚠️ Moderate confidence (Below configured {confidence_threshold}% cutoff)")
            else:
                st.error("❌ Uncertain prediction — Image may be ambiguous or out-of-distribution.")

            st.divider()
            st.subheader("📊 Top Predictions Breakdown")

            for rank in range(len(top_indices[0])):
                idx = top_indices[0][rank].item()
                score = top_probs[0][rank].item() * 100
                label = class_names[idx].replace("_", " ").title()

                col_rank, col_bar = st.columns([1, 4])
                with col_rank:
                    st.markdown(f"**#{rank + 1} {label}**")
                with col_bar:
                    st.progress(score / 100)
                    st.caption(f"{score:.2f}% confidence match")

            # CSV Export
            df_export = pd.DataFrame([{
                "Filename": single_file.name,
                "Predicted_Class": predicted_class,
                "Confidence_Pct": round(confidence, 2),
                "Threshold_Set_Pct": confidence_threshold,
                "Inference_Time_ms": round(latency, 2),
                "Device": device_name,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            
            csv_data = df_export.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Classification Report (CSV)",
                data=csv_data,
                file_name=f"classification_{single_file.name}.csv",
                mime="text/csv",
                use_container_width=True
            )

            with st.expander("🔬 Technical & Tensor Details"):
                st.write(f"**Device Assigned:** {device_name}")
                st.write("**Model Architecture:** MobileNetV3 Small")
                st.write(f"**Input Tensor Shape:** `[1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]`")
                st.write("**Normalization Mean:** `[0.485, 0.456, 0.406]`")
                st.write("**Normalization Std:** `[0.229, 0.224, 0.225]`")
                st.write(f"**Total Class Space:** {len(class_names)} categories")
                st.write(f"**Inference Latency:** `{latency:.2f} ms`")
                
                resized_preview = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                st.image(
                    resized_preview, 
                    caption=f"Network Input Tensor ({IMAGE_SIZE}x{IMAGE_SIZE} RGB)", 
                    width=150
                )

    # ------------------------------------------------------
    # BATCH MULTI-IMAGE MODE
    # ------------------------------------------------------
    else:
        st.info(f"📦 **Batch Processing Active:** {len(uploaded_files)} images selected for bulk inference.")

        run_batch = st.button(
            f"⚡ Process All {len(uploaded_files)} Images",
            use_container_width=True,
            type="primary"
        )

        if run_batch:
            batch_results = []
            progress_bar = st.progress(0)
            
            for idx, file_item in enumerate(uploaded_files):
                try:
                    img = Image.open(BytesIO(file_item.getvalue())).convert("RGB")
                    start_t = time.perf_counter()

                    img_tensor = evaluation_transform(img).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        outputs = model(img_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        top_prob, top_idx = torch.topk(probs, k=1, dim=1)

                    lat_ms = (time.perf_counter() - start_t) * 1000

                    pred_class = class_names[top_idx[0][0].item()].replace("_", " ").title()
                    conf_pct = top_prob[0][0].item() * 100
                    status = "Passed Cutoff" if conf_pct >= confidence_threshold else "Low Confidence"

                    batch_results.append({
                        "Filename": file_item.name,
                        "Predicted Category": pred_class,
                        "Confidence Score": f"{conf_pct:.2f}%",
                        "Status": status,
                        "Latency (ms)": f"{lat_ms:.2f} ms",
                        "Device": device_name,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception:
                    batch_results.append({
                        "Filename": file_item.name,
                        "Predicted Category": "Error Processing File",
                        "Confidence Score": "N/A",
                        "Status": "Error",
                        "Latency (ms)": "N/A",
                        "Device": device_name,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.divider()
            st.subheader("📊 Batch Inference Results")

            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch, use_container_width=True)

            batch_csv = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Master Batch Report (CSV)",
                data=batch_csv,
                file_name=f"batch_shoe_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================================
# ABOUT MODEL
# ==========================================================

with st.expander("🧠 About This AI Model"):
    st.markdown("""
### MobileNetV3 Small

This application uses a fine-tuned MobileNetV3
convolutional neural network trained for shoe
classification.

The model learns visual patterns including:

- Shoe shape
- Material texture
- Sole design
- Colour distribution
- Structural features

to determine the most likely shoe category.

MobileNetV3 is designed to provide strong
classification performance while remaining
lightweight and efficient.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
## 👨‍💻 Developer

**Mwila Singoma**

AKA **CodeCrafterMwila**

### Specialisations

✅ Artificial Intelligence  
✅ Deep Learning  
✅ Computer Vision  
✅ Machine Learning  
✅ Python Development

---

### Built With

✅ Python  
🧠 PyTorch  
✅ MobileNetV3  
🌐 Streamlit 

---

© 2026 CodeCrafterMwila AI Lab
""")