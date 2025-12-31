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
st.markdown("### 支援「手寫」與「語音」雙模辨識 (自動轉換阿拉伯數字)")

# --- 新增：中文/英文轉阿拉伯數字函數 ---
def convert_to_digits(text):
    if not text:
        return ""
    
    # 建立中英文對應表
    mapping = {
        '零': '0', '一': '1', '二': '2', '兩': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # 小寫化處理英文
    text = text.lower()
    
    # 1. 檢查是否包含原本就是阿拉伯數字的部分
    found_digits = re.findall(r'\d', text)
    if found_digits:
        return "".join(found_digits)
    
    # 2. 處理純中英文單字轉換
    result = ""
    # 處理連續的中文字 (例如：一二三)
    for char in text:
        if char in mapping:
            result += mapping[char]
    
    # 如果沒結果，處理英文單字 (例如：one two)
    if not result:
        words = text.split()
        for w in words:
            if w in mapping:
                result += mapping[w]
                
    return result if result else text

# --- 1. 模型與預處理邏輯 (保持不變) ---
MODEL_PATH = 'mnist_model_v2.h5'
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        model = models.Sequential([
            layers
