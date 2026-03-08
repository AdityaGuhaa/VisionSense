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
├── main.py
│
├── camera/
│   └── camera_stream.py        # Handles webcam capture
│
├── models/
│   ├── detector.py             # YOLOv8 detection wrapper
│   └── vlm.py                  # Vision‑Language model interface
│
├── utils/
│   └── drawing.py              # Bounding box and visualization utilities
│
└── requirements.txt
```

### Module Responsibilities

**camera_stream.py**
Handles real‑time frame capture from the webcam.

**detector.py**
Runs YOLOv8 inference and extracts:

* object labels
* bounding boxes
* detection confidence

**vlm.py**
Interfaces with the **Vision‑Language Model running via llama.cpp** to generate scene descriptions.

**drawing.py**
Responsible for visualization such as:

* bounding boxes
* labels
* annotations

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/visionsense.git
cd visionsense
```

## Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\\Scripts\\activate       # Windows
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

### YOLOv8

Download or load a YOLOv8 model using Ultralytics.

Example:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

### Vision‑Language Model

VisionSense uses a **Qwen2.5‑VL model running through llama.cpp**.

Steps:

1. Install and build **llama.cpp**
2. Download the **Qwen2.5‑VL GGUF model**
3. Configure the model path inside `vlm.py`

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

Future improvements planned for VisionSense:

* Temporal scene understanding (multi‑frame reasoning)
* Relationship detection between objects
* Improved prompt engineering
* Faster inference pipelines
* Edge device deployment
* Robotics integration

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
