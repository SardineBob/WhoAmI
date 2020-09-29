from cv2 import cv2
import os
from component.FaceDetect.FaceDetect import FaceDetect


# 寫入臉部辨識區塊影像方法
def writeFaceImage(faceName, faceImgSeq, sourceImg, XYWH):
    (x, y, w, h) = XYWH
    # 判斷資料夾不存在則建立
    path = os.path.join("faceImgSet", faceName)
    if os.path.exists("faceImgSet") is False:
        os.mkdir("faceImgSet")
    if os.path.exists(path) is False:
        os.mkdir(path)
    # 將臉部影像壓縮到32*32
    faceImg = sourceImg[y:y+h, x:x+w]
    faceImg = cv2.resize(faceImg, (32, 32))
    # 寫入影像檔
    cv2.imwrite(os.path.join(path, str(faceImgSeq) + ".png"), faceImg)


# 輸入這個臉部樣本的名稱
faceName = input("請輸入臉部樣本名稱：")
# 建立臉部偵測物件
faceDetect = FaceDetect()
# 讀取攝影機畫面
camera = cv2.VideoCapture(0)
# 產生畫面檢視視窗
windowName = "capture face"
cv2.namedWindow(windowName)
faceImgSeq = 0  # 臉部辨識影像編號
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
    # 針對臉部標記儲存影像
    for (x, y, w, h) in faces:
        writeFaceImage(faceName, faceImgSeq, img, (x, y, w, h))
        faceImgSeq = faceImgSeq + 1
    # 針對臉部標記做處理
    for (x, y, w, h) in faces:
        # 繪製臉部偵測標記位置
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    # 顯示畫面
    cv2.imshow(windowName, img)
    # 採集到500張，就離開
    if faceImgSeq >= 500:
        break
    # 按鍵盤的q離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉視窗
cv2.destroyAllWindows()
exit()
