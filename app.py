import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

st.set_page_config(page_title="多數字辨識 AI", layout="centered")
st.title("🔢 多數字即時辨識系統")
st.write("請在黑框內寫下一串數字（例如 123），數字之間請保持一點距離。")

# --- 模型載入與自動訓練 ---
MODEL_PATH = 'mnist_model.h5'
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)
        model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- (前面的畫板程式碼) ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 關鍵修正：確保這裡有跑辨識 ---
if canvas_result.image_data is not None:
    # 轉灰階並處理
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10: 
            digit_boxes.append((x, y, w, h))
    
    # 依照 X 座標排序
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    if len(digit_boxes) > 0:
        st.subheader("分析結果")
        results = []
        cols = st.columns(len(digit_boxes)) # 依照數字數量產生欄位
        
        for i, (x, y, w, h) in enumerate(digit_boxes):
            # 切割數字
            roi = img[y:y+h, x:x+w]
            # 加上邊框讓它更像訓練資料
            pad = 20
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            
            # 預測
            img_input = roi.reshape(1, 28, 28, 1).astype('float32') / 255
            pred = model.predict(img_input, verbose=0)
            digit = np.argmax(pred)
            results.append(str(digit))
            
            # 在網頁顯示小圖跟辨識結果
            with cols[i]:
                st.image(roi, width=60)
                st.markdown(f"### **{digit}**")
        
        # 顯示整串數字
        st.success(f"## 辨識整串數字為：{''.join(results)}")
    else:
        st.info("請在上方黑框寫字，AI 會自動偵測。")
