import glob

import cv2
import matplotlib.pyplot as plt
import pytesseract
import torch

# 1. 設定 Tesseract 執行檔路徑 (請確認與你安裝的路徑一致)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2. 載入剛剛訓練好的最佳 YOLOv5 模型
model_path = "yolov5/runs/train/anpr_model/weights/best.pt"
# 使用本地端的 yolov5 資料夾載入自訂權重
model = torch.hub.load("./yolov5", "custom", path=model_path, source="local")
model.conf = 0.5  # 設定信心度門檻 (大於 50% 才框出來)

# 3. 從驗證集中隨便挑一張圖片來測試
val_images = glob.glob("datasets/anpr/images/val/*.jpeg")
test_img_path = val_images[0]  # 你可以改索引值測試不同圖片，例如 val_images[5]

# 讀取圖片
img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. 進行 YOLO 推論
results = model(img_rgb)
detections = results.pandas().xyxy[0]  # 取得預測框座標表格

print(f"測試圖片: {test_img_path}")
print(detections)

# 5. 裁切車牌並進行 OCR 文字辨識
for _, row in detections.iterrows():
    xmin, ymin, xmax, ymax = (
        int(row["xmin"]),
        int(row["ymin"]),
        int(row["xmax"]),
        int(row["ymax"]),
    )
    conf = row["confidence"]

    # 裁切車牌區域 (ROI)
    roi = img[ymin:ymax, xmin:xmax]

    # 影像前處理：轉灰階以提升 OCR 準確率
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 使用 Pytesseract 提取文字 (psm 6 或 7 適合單行車牌文字)
    plate_text = pytesseract.image_to_string(roi_gray, config="--psm 6").strip()
    print(f"偵測到車牌 (信心度: {conf:.2f}) -> 辨識文字結果: 【 {plate_text} 】")

    # 在原圖畫上綠框與辨識出的文字
    cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    cv2.putText(
        img_rgb,
        plate_text,
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

# 6. 顯示並儲存結果圖片
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("YOLOv5 Detection + OCR Result")
plt.savefig("final_prediction_result.png", bbox_inches="tight")
plt.show()
print("結果已儲存為 final_prediction_result.png！")
