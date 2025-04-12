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


1. 📄 Dataset preparation (OCR → Q&A)
2. 🧠 Fine-tuning on LLaMA 3.2
3. 🌐 Flask web integration
4. 🎙️ Voice & emotion modules


---

## 🛠️ How This Project Was Built – Step-by-Step Guide

This project is an AI-powered Ayurvedic chatbot that combines ancient wellness wisdom with modern NLP, voice input, and emotion recognition.

---

### 📁 Step 1: Dataset Creation from Ayurvedic Texts

- **Collected PDFs** of Ayurvedic scriptures and mental health documents (🧾 ~58 texts).
- Used [`marker-pdf`](https://pypi.org/project/marker-pdf/) to **OCR** the PDFs into **Markdown files**:
  
  ```bash
  marker-pdf ./pdfs ./Data
  ```

---

### 🧠 Step 2: Generate Q&A Pairs Using an LLM

- Used a **local LLaMA 3.2 model** or `gpt-3.5-turbo` via LangChain/Ollama.
- Parsed markdowns into chunks using `Langchain` for better context.
- Auto-generated **question-answer pairs** and saved them to a CSV using:
  
  ```python
  from langchain_community.llms import Ollama
  # loop through text chunks → ask Q&A → save to qa_dataset.csv
  ```

---

### 🧪 Step 3: Fine-Tune LLaMA 3.2 on the Dataset

- Used **Unsloth + LoRA adapters** to efficiently fine-tune LLaMA-3.2-1B on Google Colab with T4 GPU.
- Uploaded Q&A dataset to Hugging Face Datasets for fast loading.
- Used `SFTTrainer` from `transformers` to perform instruction tuning.

---

### 🌐 Step 4: Build Web UI with Flask

- Built a chat interface using **Flask (backend)** and **HTML/CSS (frontend)**.
- Integrated response flow using `@app.route('/chat', methods=['POST'])`.

---

### 🎙️ Step 5: Add Voice Input and Output

- 🎤 **Speech-to-Text** using `openai/whisper`
- 🔊 **Text-to-Speech** using `pyttsx3`
- UI controls (mic/speaker buttons) were added in `index.html`.

---

### 😐 Step 6: Add Real-Time Emotion Detection

- Trained a **ResNet18 model** on FER2013 and exported to ONNX.
- Used **OpenCV** + `onnxruntime` to detect emotions live via webcam.
- Used Flask + JavaScript to show emotion in real-time and control via toggle.

---

### 📊 Step 7: Model Evaluation & Comparison

- Used `eval_emotion_onnx.py` to get:
  - Accuracy, Confusion Matrix, F1-score, Inference Time

- Used `eval_llama_model.py` to compute:
  - BLEU, ROUGE, Average Response Length

- Compared your fine-tuned model with **DistilGPT2 baseline** in `comparison.py`.

---

### 🖼️ Outputs

- 📈 Plots: Confusion Matrix, BLEU/ROUGE bar charts
- 🧠 Real-time interface with emotion-based and voice-aware responses

---


