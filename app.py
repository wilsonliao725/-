import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import streamlit.components.v1 as components
import re

# 1. 頁面基本設定
st.set_page_config(page_title="AI 數字辨識專家", layout="centered")
st.title("🔢 AI 數字辨識專家 (手寫 + 語音)")

# 2. 核心轉換函數：將中英文字轉為阿拉伯數字
def text_to_digit(text):
    if not text: return ""
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    text = text.lower()
    # 先找原本就是數字的部分
    digits = re.findall(r'\d', text)
    if digits: return "".join(digits)
    
    # 逐字/逐詞比對中英文
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    if not res:
        for word in text.split():
            if word in mapping: res += mapping[word]
    return res

# 3. 強化版 CNN 模型載入
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

# 4. 🎤 語音辨識區塊 (修正通訊邏輯)
st.subheader("🎤 語音辨識數字")

# 取得 URL 中的語音輸入值
voice_input_raw = st.query_params.get("voice_input", "")
voice_converted = text_to_digit(voice_input_raw)

# 顯示語音識別結果框 (現在會正確帶入轉換後的數字)
st.text_input("語音識別結果 (阿拉伯數字)：", value=voice_converted, key="display_box")

# JavaScript 注入 (移除 alert，改用自動導向傳值)
speech_js = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW';

function startListen() {
    const btn = document.getElementById("v_btn");
    btn.innerText = "正在聆聽中...請說話";
    btn.style.backgroundColor = "#ffa500";
    
    recognition.start();
    
    recognition.onresult = (event) => {
        const resultText = event.results[0][0].transcript;
        // 核心修改：移除 alert，直接將值塞入 URL 並自動重新整理頁面
        const url = new URL(window.location.href);
        url.searchParams.set('voice_input', resultText);
        window.location.href = url.href;
    };
    
    recognition.onerror = () => {
        btn.innerText = "辨識失敗，按我重試";
        btn.style.backgroundColor = "#ff4b4b";
    };
}
</script>
<button id="v_btn" onclick="startListen()" style="width:100%; padding:15px; background-color:#ff4b4b; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold;">
    點擊開始說話 (支援中英文數字)
</button>
"""
components.html(speech_js, height=80)

if voice_input_raw:
    st.caption(f"原始語音內容：{voice_input_raw}")

# 5. ✍️ 手寫辨識區塊 (保持優化後的邏輯)
st.write("---")
st.subheader("✍️ 手寫辨識區域")
canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 1)", stroke_width=18, stroke_color="#FFFFFF", background_color="#000000", height=300, width=600, drawing_mode="freedraw", key="canvas_expert")

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 5], key=lambda x: x[0])

    if boxes:
        final_digits = []
        for x, y, w, h in boxes:
            roi = img[y:y+h, x:x+w]
            # 置中處理
            pad = 20
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            # 預測
            pred = model.predict(roi.reshape(1, 28, 28, 1).astype('float32')/255.0, verbose=0)
            final_digits.append(str(np.argmax(pred)))
        
        st.success(f"### 手寫辨識結果：{''.join(final_digits)}")
