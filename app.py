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

# --- 畫板介面 ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=600, # 加寬畫板以便寫多個數字
    drawing_mode="freedraw",
    key="canvas",
)

# --- 多數字辨識邏輯 ---
if canvas_result.image_data is not None:
    # 轉為灰階並二值化
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 尋找輪廓 (每一個數字就是一個輪廓)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 依照 X 座標從左到右排序輪廓
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10: # 過濾掉太小的雜點
            digit_boxes.append((x, y, w, h))
    
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    if digit_boxes:
        results = []
        # 在網頁上顯示切割出的數字預覽
        cols = st.columns(len(digit_boxes))
        
        for i, (x, y, w, h) in enumerate(digit_boxes):
            # 切割數字並加上 padding (讓它更像 MNIST 格式)
            roi = img[y:y+h, x:x+w]
            pad = max(w, h) // 2
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            
            # 預測
            img_input = roi.reshape(1, 28, 28, 1).astype('float32') / 255
            pred = model.predict(img_input, verbose=0)
            digit = np.argmax
