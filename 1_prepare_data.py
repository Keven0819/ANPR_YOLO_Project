import glob
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 解析所有 XML 檔案
raw_images_dir = "raw_data/images"
xml_files = glob.glob(os.path.join(raw_images_dir, "*.xml"))

data = []
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 找出對應的圖片檔名
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    img_name = base_name + ".jpeg"
    img_path = os.path.join(raw_images_dir, img_name)

    if not os.path.exists(img_path):
        continue

    # 讀取圖片取得真實寬高
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # 取得標註框座標
    for member in root.findall("object"):
        bndbox = member.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        xmax = int(bndbox.find("xmax").text)
        ymin = int(bndbox.find("ymin").text)
        ymax = int(bndbox.find("ymax").text)

        # 轉換為 YOLO 格式 (正規化至 0~1 之間)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        data.append([img_name, img_path, x_center, y_center, w, h])

df = pd.DataFrame(
    data, columns=["img_name", "img_path", "x_center", "y_center", "w", "h"]
)
print(f"成功解析 {len(df)} 筆標註資料！")

# 2. 切分訓練集 (Train: 80%) 與驗證集 (Val: 20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# 3. 建立 YOLO 標準資料夾結構並複製檔案
def save_yolo_dataset(dataframe, folder_type):
    img_dir = os.path.join("datasets", "anpr", "images", folder_type)
    lbl_dir = os.path.join("datasets", "anpr", "labels", folder_type)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for _, row in dataframe.iterrows():
        # 複製圖片
        shutil.copy(row["img_path"], os.path.join(img_dir, row["img_name"]))

        # 建立對應的 .txt 標籤檔 (類別 0 代表 license_plate)
        txt_name = os.path.splitext(row["img_name"])[0] + ".txt"
        txt_path = os.path.join(lbl_dir, txt_name)
        with open(txt_path, "w") as f:
            f.write(
                f"0 {row['x_center']:.6f} {row['y_center']:.6f} {row['w']:.6f} {row['h']:.6f}\n"
            )


save_yolo_dataset(train_df, "train")
save_yolo_dataset(val_df, "val")
print("YOLO 格式資料集建立完成！存放於 datasets/anpr/")
