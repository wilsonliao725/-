import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import streamlit.components.v1 as components

# --- 網頁配置 (Web Page Configuration) ---
# 設定網頁標題與布局 (Set page title and layout)
st.set_page_config(page_title="AI 數字全能辨識", layout="centered")
st.title("🔢 AI 數字全能辨識系統")
st.markdown("### 支援「手寫」與「語音」雙模辨識")

# --- 1. 模型與預處理邏輯 (Model & Preprocessing Logic) ---
# 定義模型儲存路徑 (Define the path to save the model)
MODEL_PATH = 'mnist_model_v2.h5'

@st.cache_resource
def get_model():
    """
    載入或訓練模型 (Load or train the model)
    如果本地沒有模型檔，則自動使用 MNIST 資料集進行訓練並儲存
    """
    if not os.path.exists(MODEL_PATH):
        # 載入 MNIST 手寫數字資料集
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        # 資料預處理：調整維度並正規化 (Normalize data to 0-1)
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # 建立 CNN 卷積神經網路模型架構
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax') # 輸出 0-9 的機率
        ])
        
        # 編譯並訓練模型
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, verbose=0)
        # 儲存模型至本地
        model.save(MODEL_PATH)
        
    return tf.keras.models.load_model(MODEL_PATH)

# 初始化模型
model = get_model()

def pre_process_digit(roi):
    """
    影像預處理 (Image Preprocessing)
    將切割出的數字影像縮放並放入 28x28 的畫布中心，以符合 MNIST 格式
    """
    if roi.size == 0: return None, None
    h, w = roi.shape
    # 保持比例縮放到 20 像素以內 (Resize while maintaining aspect ratio)
    new_h, new_w = (20, int(20*w/h)) if h > w else (int(20*h/w), 20)
    roi_resized = cv2.resize(roi, (max(1, new_w), max(1, new_h)))
    
    # 建立 28x28 的黑色底圖
    final_img = np.zeros((28, 28), dtype=np.uint8)
    # 將縮放後的數字貼在正中央 (Place the resized image in the center)
    final_img[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = roi_resized
    
    # 回傳模型輸入格式 (1, 28, 28, 1) 與預覽圖
    return final_img.reshape(1, 28, 28, 1).astype('float32') / 255.0, final_img

# --- 2. 語音辨識功能 (Voice Recognition Function) ---
st.subheader("🎤 語音辨識數字")
st.info("點擊下方按鈕後，請對著麥克風說出數字（例如：一二三 或 One Two Three）")

# 使用 HTML/JavaScript 注入瀏覽器 Web Speech API
speech_script = """
<script>
// 初始化瀏覽器語音辨識引擎
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'zh-TW'; // 設定語言為繁體中文
recognition.interimResults = false; // 只在辨識結束後回傳結果

function startListen() {
    recognition.start(); // 開始監聽錄音
    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        // 將辨識文字傳回 Streamlit 端的 voice_input 組件
        window.parent.postMessage({type: 'streamlit:set_widget_value', value: text, key: 'voice_input'}, '*');
        // 彈出提示視窗
        alert("你說的是：" + text);
    };
}
</script>
<button onclick="startListen()" style="padding: 10px 20px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer;">
    開始語音辨識
</button>
"""
# 嵌入自定義 HTML/JS 組件
components.html(speech_script, height=70)

# 顯示語音辨識回傳的結果 (Display speech-to-text result)
voice_text = st.text_input("語音識別結果：", key="voice_result_display")

# --- 3. 手寫辨識介面 (Handwriting Recognition Interface) ---
st.subheader("✍️ 手寫辨識區域")
# 建立互動式畫板 (Create drawing canvas)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)", 
    stroke_width=18, 
    stroke_color="#FFFFFF", 
    background_color="#000000", 
    height=300, 
    width=600, 
    drawing_mode="freedraw", 
    key="canvas"
)

# 辨識邏輯 (Recognition Logic)
if canvas_result.image_data is not None:
    # 1. 取得畫板資料並轉為灰階 (Convert RGBA to Gray)
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    # 2. 影像二值化 (Binarization)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # 3. 尋找各個數字的輪廓 (Find contours of digits)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 依照 X 座標排序，確保數字從左到右排列
    digit_boxes = sorted([cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5], key=lambda b: b[0])

    if digit_boxes:
        results = []
        # 逐一處理每個偵測到的數字框
        for x, y, w, h in digit_boxes:
            # 切割出單個數字並進行預處理
            processed_input, _ = pre_process_digit(img[y:y+h, x:x+w])
            if processed_input is not None:
                # 使用 CNN 模型進行分類預測
                prediction = model.predict(processed_input, verbose=0)
                # 取得機率最高者作為辨識結果
                results.append(str(np.argmax(prediction)))
        
        # 在網頁顯示最終拼接結果 (Display concatenated results)
        st.success(f"## 手寫辨識結果：{''.join(results)}")
