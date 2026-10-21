"""RGB-Thermal Multi-Spectral Semantic Segmentation - Streamlit Web Application.

Provides interactive inference for dual-stream ResNet-18 with Cross-Modal
Attention Fusion (CMAF) trained on the MFNet dataset.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from my_model import ARCHITECTURE_ID, CustomDualStreamUNet, build_model

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DEFAULT_CHECKPOINT_PATH = os.path.join(
    "checkpoints", "resnet18_crossmodal_attention_v2_1_final", "best_checkpoint.pt"
)

MODEL_INPUT_HEIGHT = 480
MODEL_INPUT_WIDTH = 640
NUM_CLASSES = 9
IGNORE_INDEX = 255

CLASS_NAMES = [
    "Unlabeled",
    "Car",
    "Person",
    "Bike",
    "Curve",
    "Car Stop",
    "Guardrail",
    "Color Cone",
    "Bump",
]

# Verified MFNet Colormap (uint8 RGB)
MFNET_COLORMAP = np.asarray(
    [
        [0, 0, 0],        # 0: Unlabeled (Black)
        [220, 20, 60],    # 1: Car (Crimson)
        [0, 200, 0],      # 2: Person (Green)
        [30, 144, 255],   # 3: Bike (Dodger Blue)
        [255, 215, 0],    # 4: Curve (Gold)
        [255, 20, 147],   # 5: Car Stop (Deep Pink)
        [0, 255, 255],    # 6: Guardrail (Cyan)
        [255, 140, 0],    # 7: Color Cone (Dark Orange)
        [138, 43, 226],   # 8: Bump (Blue Violet)
    ],
    dtype=np.uint8,
)

# Pre-extracted demo samples available in the repository
DEMO_SAMPLES = {
    "Daytime Urban Scene (00001D)": {
        "rgb": os.path.join("demo_samples", "day_00001D_rgb.png"),
        "thermal": os.path.join("demo_samples", "day_00001D_thermal.png"),
        "description": "Daytime traffic with pedestrians, cyclists, and moving cars under standard ambient daylight.",
    },
    "Nighttime Scene (00003N)": {
        "rgb": os.path.join("demo_samples", "night_00003N_rgb.png"),
        "thermal": os.path.join("demo_samples", "night_00003N_thermal.png"),
        "description": "Low-light nighttime scene where thermal LWIR radiation reveals pedestrians invisible in RGB.",
    },
}


# ==============================================================================
# Checkpoint and LFS Pointer Verification
# ==============================================================================

def inspect_checkpoint_file(path_str: str) -> Tuple[bool, bool, int, str]:
    """Inspects a checkpoint path to verify existence, file size, and LFS pointer status.

    Returns:
        (exists, is_lfs_pointer, size_bytes, description)
    """
    path = Path(path_str)
    if not path.is_file():
        return False, False, 0, f"File does not exist at {path}"

    size = path.stat().st_size
    # Git LFS pointer files are typically ~130 bytes; real weights are ~349 MB
    if size < 1024 * 1024:  # Under 1 MB
        try:
            with open(path, "rb") as f:
                header = f.read(128)
                if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    return True, True, size, (
                        f"Git LFS pointer detected ({size} bytes). Actual weights (~349 MB) "
                        "were not pulled by git clone."
                    )
        except OSError:
            pass
        return True, False, size, f"File exists but is unusually small ({size} bytes)."

    return True, False, size, f"Valid binary checkpoint file present ({size / (1024 * 1024):.1f} MB)."


def download_checkpoint_from_url(url: str, target_path: str, progress_bar=None) -> bool:
    """Safely downloads a checkpoint from a remote URL with chunked progress."""
    import urllib.request
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        temp_target = target_path + ".tmp"

        with urllib.request.urlopen(url) as response, open(temp_target, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if progress_bar and total_size > 0:
                    progress_bar.progress(min(downloaded / total_size, 1.0))

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_target, target_path)
        return True
    except Exception as exc:
        if os.path.exists(temp_target):
            try:
                os.remove(temp_target)
            except OSError:
                pass
        st.error(f"Download failed: {exc}")
        return False


# ==============================================================================
# Model Loading & Resource Caching
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device_str: str) -> Tuple[CustomDualStreamUNet, Dict]:
    """Loads the dual-stream model and restores trained checkpoint weights."""
    device = torch.device(device_str)

    # 1. Instantiate the exact architecture without downloading ImageNet weights
    # (checkpoint contains complete weights for both encoders, fusion, and decoder)
    model = build_model(num_classes=NUM_CLASSES, in_channels=4, pretrained=False)

    # 2. Safely load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch", "N/A"),
            "best_epoch": checkpoint.get("best_epoch", "N/A"),
            "best_metric_val": checkpoint.get("best_metric_val", None),
            "architecture_id": checkpoint.get("architecture_id", "N/A"),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {"epoch": "N/A", "best_metric_val": None, "architecture_id": "N/A"}
    else:
        raise ValueError(f"Unrecognized checkpoint object format: {type(checkpoint)}")

    # Clean potential DataParallel 'module.' prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, metadata


# ==============================================================================
# Preprocessing and Postprocessing
# ==============================================================================

def preprocess_pair(
    rgb_img: Image.Image, thermal_img: Image.Image
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Applies exact training preprocessing to RGB and thermal inputs.

    Matches dataset.py:
    - Bilinear resize of RGB and Thermal to (480, 640).
    - Normalizes uint8 [0, 255] to float32 [0.0, 1.0].
    - In-model normalization handles ImageNet RGB and zero-mean thermal.
    """
    orig_w, orig_h = rgb_img.size

    # Ensure modes
    rgb_conv = rgb_img.convert("RGB")
    thermal_conv = thermal_img.convert("L")

    # Resize to model native resolution (width=640, height=480)
    rgb_resized = rgb_conv.resize(
        (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), resample=Image.Resampling.BILINEAR
    )
    thermal_resized = thermal_conv.resize(
        (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), resample=Image.Resampling.BILINEAR
    )

    rgb_arr = np.asarray(rgb_resized, dtype=np.float32) / 255.0  # [480, 640, 3]
    thermal_arr = np.asarray(thermal_resized, dtype=np.float32) / 255.0  # [480, 640]
    thermal_arr = np.expand_dims(thermal_arr, axis=-1)  # [480, 640, 1]

    # Concatenate along channel axis: [480, 640, 4] (RGB: 0:3, Thermal: 3)
    combined = np.concatenate([rgb_arr, thermal_arr], axis=-1)

    # Convert to PyTorch tensor [1, 4, 480, 640]
    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, (orig_w, orig_h)


def run_inference(
    model: CustomDualStreamUNet, tensor: torch.Tensor, orig_size: Tuple[int, int], device_str: str
) -> np.ndarray:
    """Executes forward pass without gradients and restores prediction to original resolution."""
    device = torch.device(device_str)
    orig_w, orig_h = orig_size

    with torch.no_grad():
        tensor_dev = tensor.to(device)
        logits = model(tensor_dev)  # [1, 9, 480, 640]
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [480, 640]

    # Nearest-neighbor resize back to original resolution so no invalid class IDs are created
    if (preds.shape[1], preds.shape[0]) != (orig_w, orig_h):
        preds_pil = Image.fromarray(preds, mode="L")
        preds_restored = np.array(
            preds_pil.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST),
            dtype=np.int64,
        )
    else:
        preds_restored = preds.astype(np.int64)

    return preds_restored


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Maps class indices to verified MFNet RGB colors."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        colored[mask == class_id] = MFNET_COLORMAP[class_id]
    return colored


def create_overlay(rgb_orig: Image.Image, colored_mask: np.ndarray, alpha: float) -> Image.Image:
    """Blends RGB image and colorized mask with user-defined transparency."""
    rgb_arr = np.asarray(rgb_orig.convert("RGB"), dtype=np.float32)
    mask_float = colored_mask.astype(np.float32)
    blended = np.clip(rgb_arr * (1.0 - alpha) + mask_float * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def colorize_thermal(thermal_img: Image.Image, colormap_name: str) -> Image.Image:
    """Renders grayscale thermal image with selected colormap for enhanced human perception."""
    thermal_arr = np.asarray(thermal_img.convert("L"), dtype=np.float32) / 255.0
    if colormap_name == "Grayscale":
        return thermal_img.convert("RGB")

    cmap = cm.get_cmap(colormap_name.lower())
    colored = (cmap(thermal_arr)[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(colored)


# ==============================================================================
# Streamlit UI Construction
# ==============================================================================

def main():
    st.set_page_config(
        page_title="RGB-Thermal Semantic Segmentation",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header Styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.2rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
        .badge-green { background-color: #DCFCE7; color: #166534; }
        .badge-purple { background-color: #F3E8FF; color: #6B21A8; }
        .badge-orange { background-color: #FFEDD5; color: #9A3412; }
        .legend-card {
            display: flex;
            align-items: center;
            padding: 4px 8px;
            margin-bottom: 4px;
            border-radius: 4px;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
        }
        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Compute Device
    cuda_available = torch.cuda.is_available()
    device_options = ["cuda", "cpu"] if cuda_available else ["cpu"]
    default_device = "cuda" if cuda_available else "cpu"

    # ==========================================================================
    # Sidebar
    # ==========================================================================
    with st.sidebar:
        st.title("⚙️ Model & Setup")

        # Device Selection
        selected_device = st.selectbox(
            "Inference Device",
            options=device_options,
            index=0,
            help="CPU inference runs locally or on Streamlit Cloud. CUDA provides GPU acceleration if supported.",
        )
        if selected_device == "cpu":
            st.caption("ℹ️ Running on CPU. Single image inference takes ~0.5–1.5 seconds.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            st.caption(f"🚀 CUDA GPU: {gpu_name}")

        st.divider()

        # Checkpoint Management
        st.subheader("Checkpoint Status")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value=DEFAULT_CHECKPOINT_PATH,
            help="Relative or absolute path to best_checkpoint.pt",
        )

        exists, is_lfs, file_size, desc = inspect_checkpoint_file(checkpoint_path)

        if exists and not is_lfs:
            st.success(f"✅ Ready ({file_size / (1024 * 1024):.1f} MB)")
        elif is_lfs:
            st.warning("⚠️ Git LFS Pointer Detected")
            st.info(desc)
        else:
            st.error("❌ Checkpoint Not Found")
            st.caption(desc)

        # Handling Streamlit Cloud Missing/LFS Checkpoint
        if (not exists or is_lfs):
            secret_url = None
            try:
                secret_url = st.secrets.get("CHECKPOINT_URL", None)
            except Exception:
                pass
            env_url = os.environ.get("CHECKPOINT_URL", None)
            configured_url = secret_url or env_url

            with st.expander("📥 Checkpoint Download / Retrieval", expanded=True):
                st.markdown(
                    """
                    **Deploying on Streamlit Cloud?**
                    Git LFS weights (~349 MB) are not pulled automatically by free git clones.
                    
                    **Option A (Configured Secret):**
                    Set `CHECKPOINT_URL` in your Streamlit Cloud **App Settings > Secrets**.
                    
                    **Option B (Direct URL):**
                    Provide a direct download link below (e.g., GitHub Releases asset or cloud storage).
                    """
                )
                download_url_input = st.text_input(
                    "Download URL",
                    value=configured_url if configured_url else "",
                    placeholder="https://example.com/best_checkpoint.pt",
                )

                if st.button("Download Checkpoint", type="primary", use_container_width=True):
                    if not download_url_input.strip():
                        st.error("Please enter a valid HTTP/HTTPS URL.")
                    else:
                        p_bar = st.progress(0.0)
                        with st.spinner("Downloading checkpoint weights (~349 MB)..."):
                            ok = download_checkpoint_from_url(
                                download_url_input.strip(), checkpoint_path, p_bar
                            )
                        if ok:
                            st.success("Download complete! Refreshing...")
                            st.rerun()

        st.divider()

        # Visualization Controls
        st.subheader("Visualization Settings")
        overlay_alpha = st.slider(
            "Overlay Opacity (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Adjust the blend between original RGB image and the predicted mask.",
        )

        thermal_cmap = st.selectbox(
            "Thermal False-Color Palette",
            options=["Grayscale", "Inferno", "Magma", "Plasma", "Viridis"],
            index=0,
            help="Colormap applied to the thermal channel for visualization.",
        )

        st.divider()

        # Class Legend in Sidebar
        st.subheader("Semantic Classes")
        for i, name in enumerate(CLASS_NAMES):
            color = MFNET_COLORMAP[i]
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(
                f"""
                <div class="legend-card">
                    <span class="legend-color" style="background-color: {hex_color};"></span>
                    <span style="font-size: 0.85rem; color: #1E293B;">
                        <strong>{i}:</strong> {name}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # Main Application Page
    # ==========================================================================
    st.markdown('<div class="main-header">RGB-Thermal Multi-Spectral Semantic Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Fusing visible RGB and LWIR thermal radiation using a Pretrained Dual-Stream ResNet-18 with '
        'Cross-Modal Attention Fusion (CMAF) for robust urban scene understanding.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Metadata Badges
    st.markdown(
        """
        <div>
            <span class="badge badge-blue">ResNet-18 Dual-Stream</span>
            <span class="badge badge-green">CMAF 5-Scale Fusion</span>
            <span class="badge badge-purple">MFNet Benchmark</span>
            <span class="badge badge-orange">9 Output Classes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Architecture and Motivation Summary
    with st.expander("📖 Model Architecture & Multi-Spectral Motivation", expanded=False):
        st.markdown(
            """
            ### Why RGB + Thermal Multi-Spectral Vision?
            - **RGB Limitations:** Standard 3-channel optical cameras degrade severely during pitch darkness,
              direct oncoming headlight glare, harsh shadows, and heavy fog or smoke.
            - **Thermal Strengths:** Long-Wave Infrared (LWIR, $8–14\\,\\mu\\text{m}$) sensors detect heat
              radiation emitted by living bodies, warm vehicle engines, and friction surfaces, maintaining
              crisp boundaries irrespective of ambient illumination.
            - **Cross-Modal Attention Fusion (CMAF):**
              1. **Channel Gate:** Squeeze-and-excitation MLP calculates global cross-modal dependencies.
              2. **Spatial Gate:** Joint average- and max-pooling filters localize salient thermal or RGB cues.
              3. **Residual Fusion:** Combines gated features with a linear residual skip
                 ($\\text{Fused} + 0.5 \\times [\\text{RGB} + \\text{Thermal}]$) to protect raw modality features.
            - **Trained Performance:** Val mIoU **0.5555** at epoch 22 on the MFNet split (349 MB checkpoint).
            """
        )

    # Check model readiness before loading
    model = None
    model_metadata = {}
    if exists and not is_lfs:
        try:
            with st.spinner("Loading dual-stream model weights into memory..."):
                model, model_metadata = load_model(checkpoint_path, selected_device)
        except Exception as exc:
            st.error(f"Error instantiating model or loading weights: {exc}")
    else:
        st.warning(
            "⚠️ Checkpoint weights are not currently loaded. Please review the sidebar to download "
            "or verify the checkpoint path."
        )

    st.write("")

    # ==========================================================================
    # Input Selection Modes
    # ==========================================================================
    input_tab1, input_tab2, input_tab3 = st.tabs(
        [
            "🖼️ Preloaded Demo Samples",
            "📤 Upload Paired Images (RGB + Thermal)",
            "📦 Upload Single MFNet RGBA PNG",
        ]
    )

    rgb_image: Optional[Image.Image] = None
    thermal_image: Optional[Image.Image] = None
    input_source_desc = ""

    # Tab 1: Demo Samples
    with input_tab1:
        st.markdown(
            "Select an authentic pre-registered scene pair from the MFNet validation dataset to test immediately:"
        )
        sample_choice = st.selectbox(
            "Choose Sample Scene",
            options=list(DEMO_SAMPLES.keys()),
            index=0,
        )
        sample_info = DEMO_SAMPLES[sample_choice]
        st.caption(f"**Scene context:** {sample_info['description']}")

        if os.path.isfile(sample_info["rgb"]) and os.path.isfile(sample_info["thermal"]):
            try:
                rgb_image = Image.open(sample_info["rgb"]).convert("RGB")
                thermal_image = Image.open(sample_info["thermal"]).convert("L")
                input_source_desc = sample_choice
            except Exception as exc:
                st.error(f"Failed to read demo files: {exc}")
        else:
            st.error(f"Demo files not found at expected paths: {sample_info['rgb']}")

    # Tab 2: Upload Separate Files
    with input_tab2:
        st.markdown(
            """
            Upload separate RGB and Thermal image files.
            
            > [!IMPORTANT]
            > Both images must be **spatially registered** (identical pixel dimensions and representing the exact same physical scene).
            """
        )
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_rgb = st.file_uploader(
                "Upload RGB Image (Visible)",
                type=["png", "jpg", "jpeg"],
                key="uploader_rgb",
            )
        with col_up2:
            uploaded_thermal = st.file_uploader(
                "Upload Thermal Image (Infrared / Grayscale)",
                type=["png", "jpg", "jpeg"],
                key="uploader_thermal",
            )

        if uploaded_rgb and uploaded_thermal:
            try:
                rgb_cand = Image.open(uploaded_rgb).convert("RGB")
                thermal_cand = Image.open(uploaded_thermal).convert("L")

                if rgb_cand.size != thermal_cand.size:
                    st.error(
                        f"Dimension mismatch! RGB is {rgb_cand.size} (W×H) but Thermal is {thermal_cand.size} (W×H). "
                        "The two modalities must share identical dimensions."
                    )
                else:
                    rgb_image = rgb_cand
                    thermal_image = thermal_cand
                    input_source_desc = f"Uploaded pair ({uploaded_rgb.name} & {uploaded_thermal.name})"
            except Exception as exc:
                st.error(f"Error reading uploaded files: {exc}")

    # Tab 3: Upload Single MFNet RGBA PNG
    with input_tab3:
        st.markdown(
            """
            In the benchmark MFNet dataset, samples are stored as **4-channel RGBA PNGs**:
            - Channels 0:3 = Red, Green, Blue
            - Channel 3 (Alpha) = Thermal intensity (0–255)
            """
        )
        uploaded_rgba = st.file_uploader(
            "Upload 4-channel RGBA PNG",
            type=["png"],
            key="uploader_rgba",
        )
        if uploaded_rgba:
            try:
                rgba_cand = Image.open(uploaded_rgba)
                arr = np.asarray(rgba_cand)
                if arr.ndim != 3 or arr.shape[2] != 4:
                    st.error(
                        f"Expected a 4-channel RGBA image, but received shape {arr.shape} (mode {rgba_cand.mode}). "
                        "Refusing to replace missing thermal data with zeros."
                    )
                else:
                    rgb_image = Image.fromarray(arr[:, :, :3], mode="RGB")
                    thermal_image = Image.fromarray(arr[:, :, 3], mode="L")
                    input_source_desc = f"Uploaded RGBA ({uploaded_rgba.name})"
            except Exception as exc:
                st.error(f"Error processing RGBA PNG: {exc}")

    st.write("")

    # ==========================================================================
    # Preview and Inference Execution
    # ==========================================================================
    if rgb_image is not None and thermal_image is not None:
        st.divider()
        st.subheader("🔍 Input Modalities Preview")

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.image(
                rgb_image,
                caption=f"RGB Stream ({rgb_image.width} × {rgb_image.height})",
                use_container_width=True,
            )
        with col_prev2:
            st.image(
                colorize_thermal(thermal_image, thermal_cmap),
                caption=f"Thermal Stream [{thermal_cmap}] ({thermal_image.width} × {thermal_image.height})",
                use_container_width=True,
            )

        st.write("")
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            predict_clicked = st.button(
                "🚀 Predict Segmentation",
                type="primary",
                use_container_width=True,
                disabled=(model is None),
            )

        if predict_clicked:
            if model is None:
                st.error("Cannot run inference: model weights are not loaded. See sidebar.")
                return

            with st.spinner("Preprocessing tensors & executing multi-spectral forward pass..."):
                t_start = time.perf_counter()

                # 1. Exact Preprocessing
                input_tensor, orig_dim = preprocess_pair(rgb_image, thermal_image)

                # 2. Forward inference & resolution restoration
                mask_pred = run_inference(model, input_tensor, orig_dim, selected_device)
                t_infer = (time.perf_counter() - t_start) * 1000.0

                # 3. Colorization & Overlay
                colored_mask = colorize_mask(mask_pred)
                overlay = create_overlay(rgb_image, colored_mask, overlay_alpha)

            # ==================================================================
            # Results Display
            # ==================================================================
            st.success(f"Inference completed in **{t_infer:.1f} ms** on **{selected_device.upper()}**!")

            st.subheader("📊 Segmentation Output")

            row_res1, row_res2 = st.columns(2)
            with row_res1:
                st.image(
                    colored_mask,
                    caption="Predicted Semantic Segmentation Mask",
                    use_container_width=True,
                )
            with row_res2:
                st.image(
                    overlay,
                    caption=f"RGB + Segmentation Overlay (Alpha = {overlay_alpha:.2f})",
                    use_container_width=True,
                )

            # Class Breakdown Analysis
            st.subheader("📈 Detected Class Breakdown")
            unique_classes, counts = np.unique(mask_pred, return_counts=True)
            total_pixels = mask_pred.size

            breakdown_cols = st.columns(min(len(unique_classes), 5))
            for idx, (cid, count) in enumerate(zip(unique_classes, counts)):
                col = breakdown_cols[idx % len(breakdown_cols)]
                pct = (count / total_pixels) * 100.0
                c_name = CLASS_NAMES[cid]
                color = MFNET_COLORMAP[cid]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                with col:
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 6px; background-color: #F8FAFC; border-left: 5px solid {hex_color}; margin-bottom: 8px;">
                            <div style="font-size: 0.8rem; color: #64748B;">Class {cid}</div>
                            <div style="font-weight: 700; color: #1E293B;">{c_name}</div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: #0284C7;">{pct:.2f}%</div>
                            <div style="font-size: 0.75rem; color: #94A3B8;">{count:,} px</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Download Options
            st.write("")
            down_col1, down_col2 = st.columns(2)
            with down_col1:
                mask_buf = io.BytesIO()
                Image.fromarray(colored_mask).save(mask_buf, format="PNG")
                st.download_button(
                    "💾 Download Segmentation Mask (PNG)",
                    data=mask_buf.getvalue(),
                    file_name="segmentation_mask.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with down_col2:
                overlay_buf = io.BytesIO()
                overlay.save(overlay_buf, format="PNG")
                st.download_button(
                    "💾 Download Blended Overlay (PNG)",
                    data=overlay_buf.getvalue(),
                    file_name="segmentation_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
