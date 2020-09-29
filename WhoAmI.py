import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from component.Learning.CNNModule import CNNModule
from component.FaceDetect.FaceDetect import FaceDetect
from cv2 import cv2

# 取得設計好的卷積神經網路模型
module = CNNModule()
cnnModule = module.getModel()
labels = module.labels
# 印出模型摘要
print(cnnModule.summary())
# 讀取訓練結果
cnnModule.load_weights(os.path.join("learningResult", "LearningFaceResult.h3"))
# 建立臉部偵測物件
faceDetect = FaceDetect()
# 讀取攝影機畫面
camera = cv2.VideoCapture(0)
# 產生畫面檢視視窗
windowName = "let me see you"
cv2.namedWindow(windowName)
# 播放攝影機畫面
ret, frame = camera.read()
while ret:
    ret, frame = camera.read()
    # 影像串流停止，則跳脫
    if ret is False:
        break
    # 執行臉部偵測
    img = frame.copy()
    faces = faceDetect.doDetect(img)
    # 針對臉部標記執行預測
    for (x, y, w, h) in faces:
        # 將臉部影像壓縮到32*32
        faceImg = img[y:y+h, x:x+w]
        faceImg = cv2.resize(faceImg, (32, 32))
        # 執行預測
        byteImg = np.array(faceImg)
        byteImg = byteImg.astype("float32") / 255
        listImg = np.array([byteImg])
        result = cnnModule.predict_classes(listImg)
        # 繪製臉部偵測標記位置
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # 繪製偵測文字結果(因為opencv不支援中文，這邊改用PIL畫)
        cv2.rectangle(frame, (x, y), (x+w, y-25), (0, 0, 0), -1)  # 文字的背景
        PILImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 色域從opencv的BGR改為PIL的RGB
        PILFont = ImageFont.truetype(os.path.join("resource", "NotoSansTC-Medium.otf"), 14)
        PILDraw = ImageDraw.Draw(PILImg)
        PILDraw.text((x, y-25), labels[result[0]], font=PILFont, align="left")
        frame = cv2.cvtColor(np.asarray(PILImg), cv2.COLOR_RGB2BGR)  # 色域改回來opencv的BGR
    # 顯示畫面
    cv2.imshow(windowName, frame)
    # 按鍵盤的q離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉視窗
cv2.destroyAllWindows()
exit()
