import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import streamlit.components.v1 as components
import re

# 頁面基本設定
st.set_page_config(page_title="AI 數字辨識專家", layout="centered")
st.title("🔢 AI 數字辨識專家 (手寫 + 語音)")

# --- 1. 核心轉換函數：將中英文字轉為阿拉伯數字 ---
def text_to_digit(text):
    if not text: return ""
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    # 移除標點符號並轉小寫
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    # 提取原本就是數字的部分
    digits = re.findall(r'\d', text)
    if digits: return "".join(digits)
    
    # 逐字/逐詞轉換
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    
    if not res: # 嘗試英文單詞拆分
        for word in text.split():
            if word in mapping: res += mapping[word]
            
    return res

# --- 2. 強化版 CNN 模型載入 ---
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
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, verbose=0)
        model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 3. 🎤 語音辨識區塊 (優化穩定性) ---
st.subheader("🎤 語音辨識數字")

# 使用 Session State 確保結果持久化
if "voice_result" not in st.session_state:
    st.session_state.voice_result = ""

# 語音顯示框
voice_display = st.text_input("語音識別結果 (阿拉伯數字)：", value=st.session_state.voice_result)

# JavaScript 注入：使用最新的 Web Speech 監聽
speech_js = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW';

function startListen() {
    const btn = document.getElementById("v_btn");
    btn.innerText = "正在聆聽中...請大聲說出數字";
    btn.style.backgroundColor = "#ffa500";
    
    recognition.start();
    
    recognition.onresult = (event) => {
        const resultText = event.results[0][0].transcript;
        // 透過 URL 傳值並強制重新載入，確保 Python 能夠抓到
        const url = new URL(window.location.href);
        url.searchParams.set('voice_input', resultText);
        window.parent.location.href = url.href; 
    };
    
    recognition.onerror = () => {
        btn.innerText = "辨識失敗，按我重試";
        btn.style.backgroundColor = "#ff4b4b";
    };
}
</script>
<button id="v_btn" onclick="startListen()" style="width:100%; padding:15px; background-color:#ff4b4b; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold; font-size:1.1em; margin-bottom: 10px;">
    點擊開始語音輸入 (支援一二三 / One Two Three)
</button>
"""
components.html(speech_js, height=80)

# 從 URL 獲取並更新
if st.query_params.get("voice_input"):
    raw_text = st.query_params.get("voice_input")
    converted = text_to_digit(raw_text)
    if converted != st.session_state.voice_result:
        st.session_state.voice_result = converted
        st.rerun() # 強制刷新 UI

if st.session_state.voice_result:
    st.info(f"系統偵測到數字：{st.session_state.voice_result}")

# --- 4. ✍️ 手寫辨識區塊 ---
st.write("---")
st.subheader("✍️ 手寫辨識區域")
canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 1)", stroke_width=18, stroke_color="#FFFFFF", background_color="#000000", height=300, width=600, drawing_mode="freedraw", key="canvas_v3")

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 5], key=lambda x: x[0])

    if boxes:
        final_digits = []
        for x, y, w, h in boxes:
            roi = img[y:y+h, x:x+w]
            # 置中與歸一化
            pad = 20
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            pred = model.predict(roi.reshape(1, 28, 28, 1).astype('float32')/255.0, verbose=0)
            final_digits.append(str(np.argmax(pred)))
        
        st.success(f"### 手寫辨識結果：{''.join(final_digits)}")
