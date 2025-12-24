import os
import logging
import sys
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance
from transformers import pipeline

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MODEL_ID = "facebook/mask2former-swin-tiny-coco-panoptic"
EXAMPLE_IMAGE_PATH = os.path.join("assets", "example.jpg")

# --- Model Loading ---
def load_model():
    """
    Loads the segmentation model with hardware acceleration if available.
    """
    try:
        device = 0 if torch.cuda.is_available() else -1
        if device == -1 and torch.backends.mps.is_available():
            # specialized handling for Apple Silicon if supported by the pipeline
            pass 
        
        logger.info(f"Loading model '{MODEL_ID}' on device: {'GPU' if device == 0 else 'CPU'}...")
        pipe = pipeline("image-segmentation", model=MODEL_ID, device=device)
        logger.info("Model loaded successfully.")
        return pipe
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise RuntimeError("Model loading failed. Check internet connection or GPU drivers.")

# Initialize model globally to avoid reloading per request
try:
    segment_pipeline = load_model()
except RuntimeError as e:
    segment_pipeline = None

# --- Core Logic ---

def apply_pro_filters(
    image: Image.Image, 
    mask: np.ndarray, 
    filter_type: str, 
    exposure: float, 
    contrast: float, 
    saturation: float, 
    smooth_sigma: int
) -> Image.Image:
    """Applies stylized filters to the background while preserving the masked subject."""
    
    if image is None or mask is None:
        return image

    try:
        # Resize mask if it doesn't match image (safety check)
        if mask.shape[:2] != (image.height, image.width)[::-1]:
            mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

        # Smoothing
        if smooth_sigma > 0:
            # Ensure sigma is odd for GaussianBlur if strictly using kernel size, 
            # but here we use (0,0) so sigma is calculated.
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
        
        alpha = mask.astype(float) / 255.0
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        
        img_np = np.array(image.convert("RGB"))
        bg_pil = image.convert("RGB")
        
        # Artistic Adjustments
        if exposure != 1.0:
            bg_pil = ImageEnhance.Brightness(bg_pil).enhance(exposure)
        if contrast != 1.0:
            bg_pil = ImageEnhance.Contrast(bg_pil).enhance(contrast)
        if saturation != 1.0:
            bg_pil = ImageEnhance.Color(bg_pil).enhance(saturation)
            
        bg_np = np.array(bg_pil)
        
        # Preset Filter Styles
        if filter_type == "Grayscale":
            bg_gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY)
            bg_np = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2RGB)
        elif filter_type == "Portrait Blur":
            bg_np = cv2.GaussianBlur(bg_np, (51, 51), 0)
        elif filter_type == "Deep Darken":
            bg_np = (bg_np * 0.15).astype(np.uint8)
            
        # Blending
        result = (img_np * alpha_3ch + bg_np * (1 - alpha_3ch)).astype(np.uint8)
        return Image.fromarray(result)
    except Exception as e:
        logger.error(f"Error in filter application: {e}")
        return image

def extract_transparent(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Extracts the subject with a transparent background."""
    try:
        img_rgba = image.convert("RGBA")
        if mask.shape[:2] != (image.height, image.width)[::-1]:
            mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
            
        data = np.array(img_rgba)
        data[:, :, 3] = mask
        return Image.fromarray(data)
    except Exception as e:
        logger.error(f"Error extracting transparent image: {e}")
        return image

def create_segmentation_map(image: Image.Image, results: List[Dict]) -> Image.Image:
    """Creates a visual overlay of all detected segments."""
    try:
        canvas = np.array(image.convert("RGB"))
        for res in results:
            mask = np.array(res['mask'])
            if mask.max() == 1: mask = (mask * 255).astype(np.uint8)
            
            # Generate random pastel-ish color
            color = np.random.randint(50, 255, (3,)).tolist()
            
            # Apply color overlay
            bool_mask = mask > 0
            canvas[bool_mask] = (canvas[bool_mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        return Image.fromarray(canvas)
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        return image

def generate_smart_label(labels: List[str]) -> str:
    if not labels: 
        return "Studio Focus: All"
    display_str = ', '.join(labels[:2])
    if len(labels) > 2:
        display_str += "..."
    return f"Studio Focus: {display_str}"

# --- Event Handlers ---

def handle_upload(img: Image.Image):
    if segment_pipeline is None:
        raise gr.Error("Model failed to load. Please restart the app.")
    
    if img is None: 
        return [gr.update(choices=[], value=[]), None, None, None, None, "Interactive Canvas"]
    
    logger.info("Processing new image...")
    try:
        results = segment_pipeline(img)
        labels = sorted(list(set([res['label'] for res in results])))
        ov = create_segmentation_map(img, results)
        
        # Select first label by default to trigger immediate visual feedback
        default_list = [labels[0]] if labels else []
        
        logger.info(f"Detected {len(labels)} objects.")
        return [
            gr.update(choices=labels, value=default_list), 
            results, 
            img, 
            ov, 
            None, 
            generate_smart_label(default_list)
        ]
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise gr.Error("Failed to process image. Try a different file.")

def handle_click(evt: gr.SelectData, current_labels: List[str], results: List[Dict]):
    if not results: 
        return current_labels
    
    try:
        x, y = evt.index
        new_labels = list(current_labels)
        
        for res in results:
            mask = np.array(res['mask'])
            # Check bounds just in case
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                label = res['label']
                if label in new_labels: 
                    new_labels.remove(label)
                else: 
                    new_labels.append(label)
        return new_labels
    except Exception as e:
        logger.error(f"Click handler error: {e}")
        return current_labels

def render_edit(
    labels: List[str], 
    filter_type: str, 
    exp: float, 
    con: float, 
    sat: float, 
    sm: int, 
    inv: bool, 
    results: List[Dict], 
    img: Image.Image
):
    if not img or not results: 
        return None, "Interactive Canvas", None
    
    try:
        h, w = np.array(img).shape[:2]
        m_final = np.zeros((h, w), dtype=np.uint8)
        
        if labels:
            for res in results:
                if res['label'] in labels:
                    m = np.array(res['mask'])
                    if m.max() == 1: m = (m * 255).astype(np.uint8)
                    m_final = cv2.bitwise_or(m_final, m)
        
        if inv: 
            m_final = cv2.bitwise_not(m_final)
            
        edited = apply_pro_filters(img, m_final, filter_type, exp, con, sat, sm)
        return edited, generate_smart_label(labels), m_final
    except Exception as e:
        logger.error(f"Render error: {e}")
        return img, "Error Rendering", None

def export_png(img: Image.Image, mask: np.ndarray):
    if not img or mask is None: 
        return None
    try:
        path = "exported_subject.png"
        extract_transparent(img, mask).save(path)
        return path
    except Exception as e:
        logger.error(f"Export error: {e}")
        return None

# --- UI Construction ---

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft(), title="Magic Studio") as demo:
        gr.Markdown("# 🖼️Image Segmentation Studio")
        gr.Markdown("Transform your photos using Image segmentation. Upload, Click to select objects, and Style!")
        
        # Application State
        seg_results = gr.State()
        original_img = gr.State()
        last_mask = gr.State()
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("🎨 Workspace"):
                        input_img = gr.Image(type="pil", label="Select Objects by Clicking", sources=["upload", "clipboard"])
                    with gr.TabItem("🗺️ AI Map"):
                        overlay_img = gr.Image(label="Detection Categories", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🛠️ Isolated Objects")
                label_dropdown = gr.Dropdown(choices=[], multiselect=True, label="Categories", info="Select objects to protect/highlight")
                invert_chk = gr.Checkbox(label="Invert Selection", value=False, info="Apply effects to background instead of subject")
                
                gr.Markdown("### 🌈 Filters & Effects")
                filter_selector = gr.Dropdown(
                    ["None", "Grayscale", "Portrait Blur", "Deep Darken"], 
                    label="Quick Presets", 
                    value="Grayscale"
                )
                
                with gr.Accordion("Artistic Controls", open=False):
                    exposure = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Exposure")
                    contrast = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Contrast")
                    saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Saturation")
                    smooth = gr.Slider(0, 30, value=5, label="Edge Smoothing")
                
                apply_btn = gr.Button("✨ Render Result", variant="primary")
                
            with gr.Column(scale=3):
                gr.Markdown("### 🖼️ Result")
                studio_label = gr.Markdown("**Interactive Canvas**")
                result_img = gr.Image(label="Output", type="pil", interactive=False)
                transfer_btn = gr.Button("📦 Export Transparent Subject")
                trans_output = gr.File(label="Ready for Download")

        # --- Logic & Events ---
        
        # 1. Image Upload/Change Event (Includes Example Selection)
        # This chains the analysis AND the initial render immediately.
        input_img.change(
            handle_upload, 
            inputs=[input_img], 
            outputs=[label_dropdown, seg_results, original_img, overlay_img, result_img, studio_label]
        ).then(
            render_edit,
            inputs=[label_dropdown, filter_selector, exposure, contrast, saturation, smooth, invert_chk, seg_results, original_img],
            outputs=[result_img, studio_label, last_mask]
        )

        # 2. Click Interaction
        input_img.select(
            handle_click, 
            inputs=[label_dropdown, seg_results], 
            outputs=[label_dropdown]
        )

        # 3. Manual Controls
        controls = [label_dropdown, filter_selector, exposure, contrast, saturation, smooth, invert_chk]
        apply_btn.click(
            render_edit, 
            inputs=controls + [seg_results, original_img], 
            outputs=[result_img, studio_label, last_mask]
        )
        
        # 4. Export
        transfer_btn.click(
            export_png, 
            inputs=[original_img, last_mask], 
            outputs=[trans_output]
        )

        # 5. Examples
        # Check if example exists to avoid error, otherwise allow simple string
        ex_inputs = [[EXAMPLE_IMAGE_PATH]] if os.path.exists(EXAMPLE_IMAGE_PATH) else []
        if ex_inputs:
            gr.Examples(
                examples=ex_inputs,
                inputs=[input_img],
                label="Try an Example Image"
            )
        else:
            # Fallback for online demos if file not local
            gr.Examples(
                examples=[["https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"]],
                inputs=[input_img],
                label="Try an Example Image"
            )

    return demo

if __name__ == "__main__":
    # Create assets folder if not exists
    os.makedirs("assets", exist_ok=True)
    
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
