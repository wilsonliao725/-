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
st.markdown("### 支援「手寫」與「語音」雙模辨識")

# --- 1. 模型與預處理邏輯 (保持不變) ---
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

def pre_process_digit(roi):
    if roi.size == 0: return None, None
    h, w = roi.shape
    new_h, new_w = (20, int(20*w/h)) if h > w else (int(20*h/w), 20)
    roi_resized = cv2.resize(roi, (max(1, new_w), max(1, new_h)))
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = roi_resized
    return final_img.reshape(1, 28, 28, 1).astype('float32') / 255.0, final_img

# --- 新增：中英文數字轉換邏輯 ---
def text_to_digit(text):
    if not text: return ""
    # 建立轉換字典
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    text = text.lower()
    # 處理原本就是阿拉伯數字的情況
    digits = re.findall(r'\d', text)
    if digits: return "".join(digits)
    
    # 處理逐字/逐詞轉換
    res = ""
    for char in text:
        if char in mapping: res += mapping[char]
    if not res:
        for word in text.split():
            if word in mapping: res += mapping[word]
    return res

# --- 2. 語音辨識功能 (Web Speech API) ---
st.subheader("🎤 語音辨識數字")
st.info("點擊下方按鈕後，請對著麥克風說出數字（例如：一二三 或 One Two Three）")

# 接收語音參數並轉換
voice_raw = st.query_params.get("voice_input", "")
voice_final = text_to_digit(voice_raw)

# JavaScript 腳本 (修正傳值邏輯以確保即時顯示)
speech_script = """
<script>
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW'; 
recognition.interimResults = false;

function startListen() {
    recognition.start();
    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        // 使用 window.top.location 確保繞過 iframe 限制傳值給 Streamlit
        const url = new URL(window.top.location.href);
        url.searchParams.set('voice_input', text);
        window.top.location.href = url.href;
    };
}
</script>
<button onclick="startListen()" style="padding: 15px 30px; background-color: #ff4b4b; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%;">
    開始語音辨識
</button>
"""
components.html(speech_script, height=100)

# 將轉換後的阿拉伯數字顯示在輸入框中
st.text_input("語音識別結果：", value=voice_final, key="voice_result_display")

# --- 3. 手寫辨識介面 (完全保留原始邏輯) ---
st.subheader("✍️ 手寫辨識區域")
canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 1)", stroke_width=18, stroke_color="#FFFFFF", background_color="#000000", height=300, width=600, drawing_mode="freedraw", key="canvas")

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, thresh_binary := cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_boxes = sorted([cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5], key=lambda b: b[0])

    if digit_boxes:
        results = []
        for x, y, w, h in digit_boxes:
            processed_input, _ = pre_process_digit(img[y:y+h, x:x+w])
            if processed_input is not None:
                results.append(str(np.argmax(model.predict(processed_input, verbose=0))))
        st.success(f"## 手寫辨識結果：{''.join(results)}")
