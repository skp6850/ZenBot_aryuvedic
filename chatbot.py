import torch
import numpy as np
import cv2
import onnxruntime
import threading
import time
import whisper
import pyttsx3
from flask import send_file
from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
#API WILL BE REVOKED SOON)
login('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')

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load chatbot model
tokenizer = AutoTokenizer.from_pretrained("Tweaks/qa_model_final") # Use fine-tuned model- Tweaks/qa_model_final
model = AutoModelForCausalLM.from_pretrained(
    "local_llama_model1",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)

# Load ONNX Emotion Model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#Use the huggingface Model- Tweaks/emotion_model_resnet18 
#Download the model on local machine direclty here.
onxx_session = onnxruntime.InferenceSession("Tweaks/emotion_model_resnet18", providers=["CPUExecutionProvider", "CUDAExecutionProvider"])

# Globals
current_emotion = "Neutral"
emotion_on = False
chat_history = []

# Emotion detection logic
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = np.stack([face] * 3, axis=0)
    face = np.expand_dims(face, axis=0)
    return face

def detect_emotion():
    global current_emotion, emotion_on
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        if emotion_on:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]
                input_tensor = preprocess_face(face)
                outputs = onxx_session.run(None, {"input": input_tensor})[0]
                pred = np.argmax(outputs)
                current_emotion = emotion_labels[pred]
        time.sleep(2)

    cap.release()

# Start emotion detection thread only once
threading.Thread(target=detect_emotion, daemon=True).start()

@app.route('/')
def home():
    return render_template("index.html", messages=chat_history, emotion=current_emotion, emotion_on=emotion_on)

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    user_input = request.form['user_input']
    chat_history.append({'user': user_input, 'bot': ''})

    # If greeting, force default response
    if user_input.lower().strip() in ['hi', 'hello', 'hey', 'namaste', 'hii', 'greetings','hello.']:
        response = "Hii!! I am your Ayurvedic helper."
    else:
        prompt_prefix = {
            "Angry": "User is angry. Respond calmly and suggest relaxing Ayurvedic tips:\n",
            "Sad": "User is sad. Respond supportively and with compassion:\n",
            "Fear": "User feels fearful or anxious. Respond with reassurance:\n",
            "Happy": "User is happy. Respond joyfully:\n"
        }.get(current_emotion, "User: ")

        full_prompt = f"{prompt_prefix}{user_input}\nBot:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        lines = decoded.split("\n")
        structured = "\n".join(f"\u2022 {line.strip()}" for line in lines if line.strip())
        response = f"🤖 {structured}"

    chat_history[-1]['bot'] = response
    return redirect(url_for('home'))


# Load Whisper model (load once for performance)
whisper_model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file = request.files['audio']
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    result = whisper_model.transcribe(audio_path)
    return jsonify({"text": result["text"]})

@app.route('/speak', methods=['POST'])
def speak_response():
    import pyttsx3
    import uuid
    text = request.form.get("text")
    filename = f"output_{uuid.uuid4().hex}.mp3"

    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return send_file(filename, mimetype='audio/mpeg')

@app.route('/toggle_emotion', methods=['POST'])
def toggle_emotion():
    global emotion_on
    emotion_on = not emotion_on
    return redirect(url_for('home'))

@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': current_emotion})

if __name__ == '__main__':
    app.run(debug=False)
