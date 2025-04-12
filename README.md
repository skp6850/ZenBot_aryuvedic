# ZenBot_aryuvedic


## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Login to your huggingFace account to use the models.
3. Download model files
4. Run: `python chatbot.py`


Use :
1. For emotion detection -  Tweaks/emotion_model_resnet18
2. For fine-tune Model - Tweaks/qa_model_final

## Features
- Ayurvedic chatbot
- Speech recognition
- Emotion detection
- Text-to-speech


You can just run the project on your machine directly by running chatbot.py file but incase you want to fine-tune the the model on your own all the dataset files are given.

🔁 How to Replicate This Project — ZenBot Ayurvedic Mental Health Chatbot
This guide walks you through replicating the full stack Ayurvedic AI chatbot with emotion detection and voice features from scratch.

🔧 Prerequisites
Python 3.8+

Git

A GPU-enabled system (recommended)

Conda or venv (optional but recommended)

📦 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/skp6850/ZenBot_aryuvedic.git
cd ZenBot_aryuvedic
🧪 2. Create & Activate Virtual Environment
bash
Copy
Edit
python -m venv llamaenv
source llamaenv/bin/activate  # Linux/macOS
llamaenv\Scripts\activate     # Windows
📚 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Includes: transformers, torch, flask, onnxruntime, opencv-python, pyttsx3, openai-whisper, etc.

🤖 4. Download & Prepare Models
🧠 Chatbot Model
Place the fine-tuned LLaMA model in a folder named:
local_llama_model1/

🧠 Emotion Detection Model
Make sure you have:
emotion_model_resnet18.onnx

📁 5. Prepare Dataset (Optional)
If replicating training:

Collect Ayurvedic PDFs

Use marker-pdf to OCR → convert to Markdown

Use qa_pair.py to generate Q&A

Fine-tune with unsloth or similar

🧪 6. Run the Application
bash
Copy
Edit
python chatbot.py
Go to: http://localhost:5000 in your browser
