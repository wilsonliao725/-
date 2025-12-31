import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# 設定網頁資訊
st.set_page_config(page_title="AI 數字辨識系統", layout="centered")
st.title("🔢 AI 多數字即時辨識系統")
st.write("請在黑框內寫數字，並讓數字之間保持一點距離。")

# --- 1. 定義優化預處理函數 ---
def pre_process_digit(roi):
    # 確保影像有內容
    if roi.size == 0:
        return None, None
    
    # 移除多餘空白邊界
    coords = cv2.findNonZero(roi)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = roi[y:y+h, x:x+w]
    
    # 縮放至 20x20 並保持比例
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    roi_resized = cv2.resize(roi, (max(1, new_w), max(1, new_h)))
    
    # 將其放入 28x28 的正中央 (MNIST 標準)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi_resized
    
    # 正規化
    input_data = final_img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return input_data, final_img

# --- 2. 建立/載入更強大的模型 ---
MODEL_PATH = 'mnist_model_v2.h5'
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('正在訓練強化版模型...'):
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
                layers.Dropout(0.2), # 防止過擬合
                layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 3. 畫板介面 ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)", # 這裡補齊了引號
    stroke_width=15, 
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=600,
    drawing_mode="freedraw",
    key="canvas_v2",
)

# --- 4. 辨識邏輯 ---
if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 4 and h > 8: 
            digit_boxes.append((x, y, w, h))
    
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    if digit_boxes:
        st.write("---")
        results = []
        cols = st.columns(len(digit_boxes))
        
        for i, (x, y, w, h) in enumerate(digit_boxes):
            roi = img[y:y+h, x:x+w]
            processed_input, preview_img = pre_process_digit(roi)
            
            if processed_input is not None:
                pred = model.predict(processed_input, verbose=0)
                digit = np.argmax(pred)
                results.append(str(digit))
                
                with cols[i]:
                    st.image(preview_img, width=60)
                    st.markdown(f"<h3 style='text-align: center;'>{digit}</h3>", unsafe_allow_html=True)
