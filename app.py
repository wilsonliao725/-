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
st.set_page_config(page_title="AI 數位全能發票系統", layout="centered")
st.title("🔢 AI 數位全能發票系統")
st.markdown("### 支援手寫與語音轉換阿拉伯數字")

# 2. 核心轉換函數：處理中英文字轉數字
def text_to_digit(text):
    if not text: return ""
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    text = text.lower()
    # 提取原本就是數字的部分
    digits = re.findall(r'\d', text)
    if digits: return "".join(digits)
    
    # 逐字比對中英文
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    if not res:
        for w in text.split():
            if w in mapping: res += mapping[w]
    return res

# 3. 強化版模型載入 (解決 ValueError)
@st.cache_resource
def get_model():
    if not os.path.exists('mnist_final.h5'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, verbose=0)
        model.save('mnist_final.h5')
    return tf.keras.models.load_model('mnist_final.h5')

model = get_model()

# 4. 🎤 語音辨識區塊 (解決沒反應問題)
st.subheader("🎤 語音辨識數字")

# 初始化 Session State 來存放語音結果
if "voice_final" not in st.session_state:
    st.session_state.voice_final = ""

# 顯示阿拉伯數字框
st.text_input("語音識別結果 (阿拉伯數字)：", value=st.session_state.voice_final, key="voice_box")

# 這裡改用 st.query_params 直接監控 URL 的變化
q_v = st.query_params.get("v", "")
if q_v:
    converted = text_to_digit(q_v)
    if converted != st.session_state.voice_final:
        st.session_state.voice_final = converted
        st.rerun() # 強制刷新畫面

# JavaScript 邏輯：強制頂層刷新 URL
speech_js = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW';

function runSpeech() {
    const btn = document.getElementById("vbtn");
    btn.innerText = "正在聆聽中...說完會自動載入";
    btn.style.backgroundColor = "#ffa500";
    
    recognition.start();
    
    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        // 核心修改：使用 window.top 確保繞過 Streamlit 的 iframe 限制
        const url = new URL(window.top.location.href);
        url.searchParams.set('v', text);
        window.top.location.href = url.href;
    };
    
    recognition.onerror = (e) => {
        btn.innerText = "失敗，請再按一次";
        btn.style.backgroundColor = "#ff4b4b";
    };
}
</script>
<button id="vbtn" onclick="runSpeech()" style="width:100%; padding:15px; background-color:#ff4b4b; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold;">
    🎤 點擊開始語音輸入 (說完自動填入)
</button>
"""
components.html(speech_js, height=90)

# 5. ✍️ 手寫辨識區域 (截圖中已確認此部分正常)
st.write("---")
st.subheader("✍️ 手寫辨識區域")
canv = st_canvas(fill_color="white", stroke_width=18, stroke_color="white", background_color="black", height=300, width=600, key="expert_v1")

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
            p = model.predict(roi.reshape(1,28,28,1).astype('float32')/255.0, verbose=0)
            final.append(str(np.argmax(p)))
        st.success(f"### 手寫辨識結果：{''.join(final)}")
