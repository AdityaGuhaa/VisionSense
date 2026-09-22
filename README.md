# VisionSense

<img width="1920" height="1039" alt="visionsense" src="https://github.com/user-attachments/assets/f8e2d76b-59e2-4574-9374-089cf2efe710" />

---

**VisionSense** is a real‑time multimodal perception system that combines **computer vision** and **vision‑language models (VLMs)** to transform raw visual input into meaningful natural‑language scene understanding.

The project integrates **YOLOv8 object detection** with a **Vision‑Language Model (Qwen2.5‑VL running via llama.cpp)** to analyze live webcam frames and generate human‑readable descriptions of the scene.

Instead of only detecting objects, VisionSense aims to move toward **true scene understanding**, bridging the gap between **perception (seeing)** and **reasoning (understanding)**.

---

# Features

### Real‑Time Scene Perception

Processes live webcam frames and performs object detection in real time.

### Object Detection with YOLOv8

Uses the **Ultralytics YOLOv8 model** to detect objects and generate bounding boxes.

### Vision‑Language Scene Understanding

Detected objects and scene context are passed to a **Vision‑Language Model (Qwen2.5‑VL)** which generates natural language descriptions.

### Modular System Architecture

Each component is separated into clean modules for maintainability and extensibility.

### Real‑Time Visual Pipeline

```
Camera → Frame Capture
      ↓
YOLOv8 Object Detection
      ↓
Bounding Boxes + Labels
      ↓
Vision‑Language Model (Qwen2.5‑VL via llama.cpp)
      ↓
Natural Language Scene Description
```

### Designed for Experimentation

The system is designed to help developers and researchers explore:

* Multimodal AI systems
* Real‑time perception pipelines
* Vision + language integration
* Robotics perception

---

# Motivation

Traditional computer vision systems stop at **object detection or classification**.

However, intelligent systems such as **robots, assistive technologies, and autonomous agents** require deeper understanding of scenes.

VisionSense explores a pipeline that moves from:

```
Pixels → Objects → Context → Language
```

By combining **detection models** with **multimodal LLMs**, the system can produce **semantic descriptions of visual environments**, opening the door to more intelligent AI agents.

---

# System Architecture

The system is structured as a modular pipeline.

```
VisionSense
│
├── main.py                     # Main application loop
├── config.py                   # Centralized configuration (camera, intervals, thresholds)
│
├── camera/
│   └── camera_stream.py        # Handles webcam capture & frame streaming
│
├── models/
│   ├── detector.py             # YOLOv8 object detection wrapper
│   └── vlm.py                  # Vision‑Language model interface (spawns persistent llama-server)
│
├── prompts/
│   └── scene_prompt.txt        # Prompts used for multimodal reasoning
│
├── utils/
│   ├── drawing.py              # Bounding box and text visualization utilities
│   └── fps.py                  # Real-time FPS calculation & display
│
└── requirements.txt
```

### Module Responsibilities

**camera_stream.py**
Handles real‑time frame capture from the webcam.

**config.py**
Centralized settings for camera index, resolution, frame intervals, and UI window parameters.

**detector.py**
Runs YOLOv8 inference and extracts:
* object labels
* bounding boxes
* detection confidence

**vlm.py**
Interfaces with the **Vision‑Language Model** via a background `llama-server` process to achieve low-latency scene descriptions without reloading model weights on each frame.

**drawing.py**
Responsible for visualization such as:
* bounding boxes
* labels
* annotations

**fps.py**
Tracks and displays real-time frame rates.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/AdityaGuhaa/VisionSense.git
cd VisionSense
```

## Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate       # Windows
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include:

* Python 3.9+
* OpenCV
* Ultralytics YOLOv8
* NumPy
* llama.cpp

---

# Model Setup

### 1. YOLOv8
The base model file (`yolov8n.pt`) is downloaded automatically by Ultralytics or included directly in the root directory.

### 2. Vision‑Language Model (Qwen2.5‑VL)

VisionSense uses **Qwen2.5‑VL‑3B‑Instruct** in GGUF format with `llama-server`.

1. **Install and build `llama.cpp`** with CUDA support:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ```
2. **Download Model & Multimodal Projector (`mmproj`)**:
   Create the directory `models/qwen_vl/` inside VisionSense and download the required weights:
   ```bash
   mkdir -p models/qwen_vl
   cd models/qwen_vl
   wget https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
   wget https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
   ```
3. **Configure Paths**:
   Ensure `self.cli_path` in `models/vlm.py` points to your built `llama-server` binary (e.g. `/path/to/llama.cpp/build/bin/llama-server`).

---

# Running VisionSense

Start the real‑time pipeline:

```bash
python main.py
```

The system will:

1. Open the webcam
2. Detect objects using YOLOv8
3. Send the frame to the Vision‑Language Model
4. Generate a natural language scene description

---

# Example Output

Example generated description:

> "A person is standing in front of a desk with a laptop. A chair is placed beside the table and a monitor is visible in the background."

This demonstrates the transition from **simple object detection to contextual scene understanding**.

---

# Applications

VisionSense can be used as a base system for multiple AI applications:

### Robotics Perception

Robots can interpret their surroundings using natural language descriptions.

### Assistive Technology

Helps visually impaired users understand their environment.

### Smart Surveillance

Provides semantic interpretation of surveillance footage.

### Human‑AI Interaction

Allows machines to describe what they "see".

### Research in Multimodal AI

Useful for experimenting with **vision + language architectures**.

---

# Roadmap

* [x] Faster inference pipeline (persistent background `llama-server` streaming)
* [ ] Temporal scene understanding (multi‑frame reasoning)
* [ ] Relationship detection between objects
* [ ] Improved prompt engineering & context pruning
* [ ] Edge device deployment
* [ ] Robotics integration

---

# Technologies Used

* **Python**
* **OpenCV**
* **Ultralytics YOLOv8**
* **Vision‑Language Models**
* **llama.cpp**
* **Computer Vision**
* **Multimodal AI**

---

# Contributing

Contributions are welcome!

If you would like to improve VisionSense:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

# License

This project is released under the **MIT License**.

---

# Author

**Aditya Guha**
AI & Machine Learning Enthusiast
Computer Science Engineering (AI & ML)

Exploring the intersection of **Computer Vision, Robotics, and Multimodal AI Systems**.

---

⭐ If you found this project interesting, consider starring the repository!
