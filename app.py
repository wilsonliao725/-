import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import streamlit.components.v1 as components
import re

# 頁面設定
st.set_page_config(page_title="AI 數字全能辨識", layout="centered")
st.title("🔢 AI 數字全能辨識系統")
st.markdown("### 支援手寫與語音轉換阿拉伯數字")

# --- 1. 核心轉換函數：將中英文轉為阿拉伯數字 ---
def convert_to_digits(text):
    if not text: return ""
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    text = text.lower()
    # 提取原本就是阿拉伯數字的部分
    found = re.findall(r'\d', text)
    if found: return "".join(found)
    
    # 逐字/詞轉換中英文
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    if not res:
        for w in text.split():
            if w in mapping: res += mapping[w]
    return res

# --- 2. 強化版模型載入 (已修正 metrics 語法) ---
MODEL_PATH = 'mnist_model_final.h5'
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
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        # 修正：metrics 必須是列表 ['accuracy']
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3, verbose=0)
        model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 3. 語音辨識功能 ---
st.subheader("🎤 語音辨識數字")

# 獲取傳回的語音參數
raw_voice = st.query_params.get("v", "")
voice_res = convert_to_digits(raw_voice)

# 顯示框框並自動填入轉換後的數字
st.text_input("語音識別結果 (阿拉伯數字)：", value=voice_res)

# JavaScript 邏輯：移除跳轉彈窗，直接將值帶入 URL
js_code = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW';
function start() {
    const btn = document.getElementById("vbtn");
    btn.innerText = "聆聽中...";
    recognition.start();
    recognition.onresult = (e) => {
        const t = e.results[0][0].transcript;
        const u = new URL(window.location.href);
        u.searchParams.set('v', t);
        window.parent.location.href = u.href;
    };
}
</script>
<button id="vbtn" onclick="start()" style="width:100%; padding:15px; background-color:#ff4b4b; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold;">
    🎤 開始語音輸入
</button>
"""
components.html(js_code, height=80)

if raw_voice:
    st.caption(f"原始辨識內容：{raw_voice}")

# --- 4. 手寫辨識區域 ---
st.write("---")
st.subheader("✍️ 手寫辨識區域")
canvas_result = st_canvas(fill_color="white", stroke_width=18, stroke_color="white", background_color="black", height=300, width=600, drawing_mode="freedraw", key="canvas_final")

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, th = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 5], key=lambda x: x[0])

    if boxes:
        final_res = []
        for x, y, w, h in boxes:
            roi = img[y:y+h, x:x+w]
            # 置中與大小調整
            roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            # 預測
            p = model.predict(roi.reshape(1, 28, 28, 1).astype('float32')/255.0, verbose=0)
            final_res.append(str(np.argmax(p)))
        st.success(f"### 手寫辨識結果：{''.join(final_res)}")
