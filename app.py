import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

st.set_page_config(page_title="多數字辨識優化版", layout="centered")
st.title("🔢 AI 多數字即時辨識系統")
st.write("優化了 4 與 9 的辨識率，請在下方書寫。")

# --- 1. 定義優化預處理函數 (修正重心偏移) ---
def pre_process_digit(roi):
    # 縮放到 20x20，保留邊界
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    roi_resized = cv2.resize(roi, (new_w, new_h))
    
    # 將 20x20 放入 28x28 的中心
    final_img = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi_resized
    
    # 正規化
    input_data = final_img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return input_data, final_img

# --- 2. 模型載入 (提高訓練輪數以精準辨識 4/9) ---
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
            layers.Conv2D(64, (3, 3), activation='relu'), # 增加一層提高特徵抓取
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0) # 提高到 5 輪
        model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 3. 畫板介面 (稍微加粗筆觸) ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1
