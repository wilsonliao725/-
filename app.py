import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# 設定網頁標題與樣式
st.set_page_config(page_title="AI 手寫數字辨識", layout="centered")
st.title("🔢 手寫數字辨識 AI 網站")
st.write("在下方黑框寫一個 0-9 的數字，AI 會即時辨識！")

# --- 模型檢查與訓練邏輯 ---
MODEL_PATH = 'mnist_model.h5'

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('首次執行，正在訓練 AI 模型，請稍候約 30 秒...'):
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
            model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
            model.save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = get_model()

# --- 畫板介面 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("手寫區域")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )

with col2:
    st.subheader("辨識結果")
    if canvas_result.image_data is not None:
        # 影像預處理
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img_input = img.reshape(1, 28, 28, 1).astype('float32') / 255
        
        # 進行預測
        prediction = model.predict(img_input)
        final_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 顯示大大的結果
        st.metric(label="預測數字", value=str(final_digit))
        st.write(f"信心度：{confidence:.2%}")
        st.progress(float(confidence))
