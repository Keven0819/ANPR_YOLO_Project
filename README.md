# Automatic Number Plate Recognition (ANPR) with YOLOv5

本專案復刻自 Kaggle 經典專案 [Automatic Number Plate Recognition](https://www.kaggle.com/code/aslanahmedov/automatic-number-plate-recognition)。
目標是將原始的 PASCAL VOC (`.xml`) 車牌資料集轉換為 YOLO 格式，並於本機端使用 GPU 訓練 **YOLOv5** 物件偵測模型，最後結合 **Tesseract-OCR** 達成車牌定位與字元辨識的完整流程。

## 💻 硬體與環境配置
* **OS**: Windows 11
* **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
* **Python**: 3.10+ (使用 `.venv` 虛擬環境)
* **Framework**: PyTorch (CUDA 12.1), YOLOv5
* **OCR Engine**: Tesseract-OCR

## 📂 專案目錄結構
```text
ANPR_YOLO_Project/
├── .venv/                     # Python 虛擬環境
├── raw_data/                  # 解壓縮後的 Kaggle 原始資料集
│   ├── images/                # 原始 .jpeg 圖片與 .xml 標註檔
│   └── TEST/                  # 測試用圖片與影片
├── datasets/                  # 轉換後的 YOLO 格式資料集 (由腳本自動生成)
│   └── anpr/
│       ├── images/            # train (80%) & val (20%)
│       └── labels/            # 轉換後的正規化座標 .txt 檔
├── yolov5/                    # Ultralytics YOLOv5 官方開源庫
├── 1_prepare_data.py          # 資料前處理腳本 (XML 轉 YOLO TXT)
├── 2_predict_and_ocr.py       # 模型推論與 OCR 字元辨識腳本
├── dataset.yaml               # YOLO 訓練設定檔
└── README.md
```

## 🚀 快速開始 (Quick Start)

### 1. 建立環境與安裝依賴
```bash
# 建立並啟動虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝支援 CUDA 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 下載 YOLOv5 並安裝相關套件
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
pip install pandas scikit-learn pytesseract opencv-python matplotlib
```
*(註：系統需另外安裝 [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) 執行檔，預設路徑為 `C:\Program Files\Tesseract-OCR\tesseract.exe`)*

### 2. 資料準備與格式轉換
前往 [Kaggle Dataset](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection) 下載資料集，解壓縮至 `raw_data/` 目錄下，接著執行轉換腳本：
```bash
python 1_prepare_data.py
```

### 3. 模型訓練 (Training)
使用本機 GPU 訓練 YOLOv5s 模型（100 epochs）：
```bash
python yolov5/train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --device 0 --workers 2 --name anpr_model
```
訓練結果與最佳權重檔將儲存於 `yolov5/runs/train/anpr_model/weights/best.pt`。

### 4. 推論與車牌字元辨識 (Inference & OCR)
載入訓練好的最佳權重，對驗證集圖片進行車牌框選並透過 Pytesseract 讀取車牌號碼：
```bash
python 2_predict_and_ocr.py
```

## 📊 訓練成果
* 模型評估指標（mAP、Loss 曲線、混淆矩陣）位於 `yolov5/runs/train/anpr_model/`。
* 最終結合 OCR 之預測輸出圖片將儲存為根目錄下的 `final_prediction_result.png`。
