import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import os
import streamlit.components.v1 as components
import re

st.set_page_config(page_title="AI Multi-Digit", layout="centered")
st.title("🔢 AI All-in-One Recognition")

# --- 1. Digit Converter ---
def text_to_digit(text):
    if not text: return ""
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    text = text.lower()
    digits = re.findall(r'\d', text)
    if digits: return "".join(digits)
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    if not res:
        for w in text.split():
            if w in mapping: res += mapping[w]
    return res

# --- 2. Load Model ---
@st.cache_resource
def get_model():
    if not os.path.exists('model.h5'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, verbose=0)
        model.save('model.h5')
    return tf.keras.models.load_model('model.h5')

model = get_model()

# --- 3. Voice Logic ---
st.subheader("🎤 Voice Recognition")

# Capture from URL
voice_raw = st.query_params.get("v", "")
voice_res = text_to_digit(voice_raw)

st.text_input("Converted Digit:", value=voice_res)

# JS logic with pure English comments to avoid SyntaxError
js_code = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW';
function start() {
    const b = document.getElementById("b");
    b.innerText = "Listening...";
    recognition.start();
    recognition.onresult = (e) => {
        const t = e.results[0][0].transcript;
        const u = new URL(window.location.href);
        u.searchParams.set('v', t);
        window.parent.location.href = u.href;
    };
}
</script>
<button id="b" onclick="start()" style="width:100%; padding:15px; background-color:#ff4b4b; color:white; border:none; border-radius:10px; cursor:pointer;">
    Click to Speak
</button>
"""
components.html(js_code, height=80)

if voice_raw:
    st.write(f"Raw Input: {voice_raw}")

# --- 4. Handwriting Logic ---
st.write("---")
st.subheader("✍️ Handwriting Area")
canv = st_canvas(fill_color="white", stroke_width=15, stroke_color="white", background_color="black", height=300, width=600, key="canvas")

if canv.image_data is not None:
    img = cv2.cvtColor(canv.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, th = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 5], key=lambda x: x[0])
    
    if boxes:
        final = []
        for x,y,w,h in boxes:
            roi = img[y:y+h, x:x+w]
            roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            p = model.predict(roi.reshape(1,28,28,1)/255.0, verbose=0)
            final.append(str(np.argmax(p)))
        st.success(f"Result: {''.join(final)}")
