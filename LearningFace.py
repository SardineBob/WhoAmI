import os
import numpy as np
from keras.utils import np_utils
from component.Learning.CNNModule import CNNModule
from PIL import Image

# 取得設計好的卷積神經網路模型
cnnModule = CNNModule().getModel()
# 印出模型摘要
print(cnnModule.summary())
# 準備訓練樣本
ImgList = []
LabList = []
for rootPath, path, files in os.walk("faceImgSet"):
    if(len(files) > 0):
        for imgFile in files:
            # 讀取圖片
            img = Image.open(os.path.join(rootPath, imgFile))
            byteImg = np.array(img)  # 轉位元陣列
            byteImg = byteImg.astype("float32") / 255  # 將值轉換成0到1之間
            ImgList.append(byteImg)
            # 放置Label(正確答案)
            if "Bob" in rootPath:
                LabList.append([0])
            if "None" in rootPath:
                LabList.append([1])

# 影像訓練樣本(轉成四層階陣列，1000張32*32擁有RGB深度3的樣本)
templates = np.array(ImgList)
# 影像樣本真實答案
labels = np.array(LabList)
labels = np_utils.to_categorical(labels)

# 設定訓練方法
cnnModule.compile(
    loss="categorical_crossentropy",  # 損失函式，通常會使用cross_entropy交叉諦，訓練效果佳
    optimizer="adam",  # 優化方法，通常會使用adam最優化方法，可讓訓練更快收斂，提高準確率
    metrics=["accuracy"]  # 評估模型方法使用accuracy準確率
)
# 開始訓練
result = cnnModule.fit(
    templates,  # 經過標準化處理的影像特徵值
    labels,  # 影像真實的值(我理解成正確的分類)，這個經過One-hot encoding處理
    validation_split=0.2,  # 驗證資料的比例，例如50000筆資料，會拿50000*0.8=40000筆資料訓練，10000筆資料驗證
    epochs=10,  # 執行10次訓練週期
    batch_size=100,  # 每批100筆資料
    verbose=1  # 顯示訓練過程
)
# 儲存訓練結果
cnnModule.save_weights(os.path.join("learningResult", "LearningFaceResult.h3"))
# 檢視訓練結果
print(result)
