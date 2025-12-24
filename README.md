# image-segmentation-using-transformers

image-segmentation-using-transformers is an image editing app built with **Gradio** and **Hugging Face Transformers**.  
It lets you **click on objects in an image, isolate them using semantic segmentation, and apply professional background filters** — all in your browser.

---
[Demo](demo.png)

## ✨ Features

- 🧠 **Object Segmentation**
  - Powered by `facebook/mask2former-swin-tiny-coco-panoptic`
  - Automatically detects and segments multiple objects in an image

- 🖱️ **Click-to-Select Objects**
  - Click directly on the image to toggle object categories
  - Multi-object selection supported

- 🎨 **Filters**
  - Grayscale
  - Portrait Blur
  - Deep Darken
  - Adjustable exposure, contrast, saturation, and edge smoothing

- 🔄 **Invert Selection**
  - Apply effects to the background instead of the subject

- 🗺️ **Segmentation Map View**
  - Visual overlay of all detected objects

- 📦 **Transparent PNG Export**
  - Export isolated subjects with alpha transparency

---

## 🚀 Demo UI

The app is built with **Gradio Blocks** and runs locally in your browser.

**Workspace Tabs**
- 🎨 Workspace – interactive object selection
- 🗺️ AI Map – category visualization overlay

**Output**
- Live-rendered result image
- Downloadable transparent PNG

---

## 🧠 Model

- **Model ID:** `facebook/mask2former-swin-tiny-coco-panoptic`
- **Task:** Image Segmentation (Panoptic)
- **Backend:** Hugging Face `pipeline`
- **Hardware Acceleration:**  
  - CUDA (NVIDIA GPUs)
  - CPU fallback (automatic)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/magic-segmentation-studio.git
cd magic-segmentation-studio
````

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages include:**

* gradio
* torch
* transformers
* pillow
* numpy
* opencv-python

---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser at:

```
http://localhost:7860
```

---

## 🖼️ Example Image

* Place an example image at:

  ```
  assets/example.jpg
  ```
* If not found, the app automatically falls back to an online demo image.

---

## 🧩 How It Works

1. **Upload Image**

   * Image is passed to the segmentation pipeline

2. **Segmentation**

   * Each detected object returns:

     * label
     * binary mask

3. **User Interaction**

   * Click image → toggle labels
   * Select labels via dropdown

4. **Mask Processing**

   * Masks are merged, smoothed, and optionally inverted

5. **Rendering**

   * Filters applied only to background or subject
   * Final image blended using alpha masks

6. **Export**

   * Subject extracted as transparent PNG

---

## 🛠️ Project Structure

```
.
├── app.py                # Main application
├── assets/
│   └── example.jpg       # Optional example image
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes & Limitations

* First run will download the model (~hundreds of MB)
* Performance depends on GPU availability
* Apple Silicon uses CPU unless supported by the pipeline
* Designed for experimentation and creative workflows

---

## 📜 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it.

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* Facebook AI Research (Mask2Former)
* Gradio Team

---
