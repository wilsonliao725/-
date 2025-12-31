import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import streamlit.components.v1 as components
import re

# 設定網頁資訊
st.set_page_config(page_title="AI 數字全能辨識", layout="centered")
st.title("🔢 AI 數字全能辨識系統")
st.markdown("### 支援「手寫」與「語音」雙模辨識 (自動轉阿拉伯數字)")

# --- 1. 數字轉換器函數 ---
def text_to_digit(text):
    # 建立轉換字典
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    # 轉小寫處理英文
    text = text.lower()
    
    # 1. 先處理直接是數字的情況 (例如 "123")
    digits_only = re.findall(r'\d', text)
    if digits_only:
        return "".join(digits_only)
    
    # 2. 處理中英文單字 (例如 "一二三" 或 "one two")
    result = ""
    # 簡單的逐字/逐詞比對
    for char in text:
        if char in mapping:
            result += mapping[char]
            
    # 如果逐字比對沒結果，嘗試英文單詞拆分比對
    if not result:
        words = text.split()
        for w in words:
            if w in mapping:
                result += mapping[w]
                
    return result if result else text

# --- 2. 模型載入邏輯 (保持不變) ---
MODEL_PATH = 'mnist_model_v2.h5'
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, verbose=0)
        model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 3. 語音辨識功能與 JavaScript 注入 ---
st.subheader("🎤 語音辨識數字")

# 透過 query_params 來接收 JavaScript 傳回的值
if "voice_output" not in st.session_state:
    st.session_state.voice_output = ""

# JavaScript 修正版：使用 window.location.href 或 Streamlit 原生回傳機制
speech_script = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW'; 
recognition.interimResults = false;

function startListen() {
    const btn = document.getElementById("record_btn");
    btn.innerText = "正在聆聽中...";
    btn.style.backgroundColor = "#ffaa00";
    
    recognition.start();
    
    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        // 透過 Streamlit 的方式將值傳回
        const link = document.createElement('a');
        link.href = `?voice_input=${encodeURIComponent(text)}`;
        link.click();
    };
    
    recognition.onerror = (event) => {
        alert("語音辨識發生錯誤: " + event.error);
        btn.innerText = "開始語音辨識";
        btn.style.backgroundColor = "#ff4b4b";
    };
}
</script>
<button id="record_btn" onclick="startListen()" style="padding: 15px 30px; background-color: #ff4b4b; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%;">
    按我開始說話
</button>
"""
components.html(speech_script, height=100)

# 獲取從 URL 傳回的語音文字
query_params = st.query_params
raw_voice = query_params.get("voice_input", "")

if raw_voice:
    converted_digit = text_to_digit(raw_voice)
    st.success(f"辨識到的原始語音：{raw_voice}")
    st.text_input("語音識別結果 (阿拉伯數字)：", value=converted_digit, key="voice_display")
else:
    st.text_input("語音識別結果 (阿拉伯數字)：", value="", key="voice_display_empty")

# --- 4. 手寫辨識介面 (保持之前優化過的邏輯) ---
st.write("---")
st.subheader("✍️ 手寫辨識區域")
canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 1)", stroke_width=18, stroke_color="#FFFFFF", background_color="#000000", height=300, width=600, drawing_mode="freedraw", key="canvas")

def pre_process_digit(roi):
    if roi.size == 0: return None
    h, w = roi.shape
    new_h, new_w = (20, int(20*w/h)) if h > w else (int(20*h/w), 20)
    roi_resized = cv2.resize(roi, (max(1, new_w), max(1, new_h)))
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = roi_resized
    return final_img.reshape(1, 28, 28, 1).astype('float32') / 255.0

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_boxes = sorted([cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5], key=lambda b: b[0])

    if digit_boxes:
        results = []
        for x, y, w, h in digit_boxes:
            processed_input = pre_process_digit(img[y:y+h, x:x+w])
            if processed_input is not None:
                results.append(str(np.argmax(model.predict(processed_input, verbose=0))))
        st.success(f"## 手寫辨識結果：{''.join(results)}")
