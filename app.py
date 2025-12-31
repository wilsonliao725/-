import cv2
import numpy as np

# ... (前面的 model 載入程式碼保持不變) ...

if canvas_result.image_data is not None:
    # 1. 影像預處理
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    
    # 2. 尋找數字輪廓 (Contours)
    # 使用 cv2.RETR_EXTERNAL 只找最外層輪廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 依照 X 座標排序，確保數字是從左到右辨識
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    results = []
    
    for cnt in contours:
        # 取得每個數字的邊框
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 忽略太小的雜點
        if w > 5 and h > 5:
            # 切割出單個數字並加入填充 (Padding) 讓辨識更準
            digit_img = img[y:y+h, x:x+w]
            digit_img = cv2.copyMakeBorder(digit_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            
            # 縮放成模型要的 28x28
            digit_resized = cv2.resize(digit_img, (28, 28))
            
            # 模型預測
            img_input = digit_resized.reshape(1, 28, 28, 1).astype('float32') / 255
            prediction = model.predict(img_input)
            results.append(str(np.argmax(prediction)))

    # 顯示最終結果
    if results:
        final_number = "".join(results)
        st.header(f"辨識結果：{final_number}")
    else:
        st.write("請在畫板上寫字")
