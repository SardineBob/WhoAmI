from cv2 import cv2


# 臉部辨識類別
class FaceDetect():

    __faceDetectClass = None

    def __init__(self):
        # 載入人臉辨識分類器
        self.__faceDetectClass = cv2.CascadeClassifier("component\FaceDetect\haarcascade_frontalface_default.xml")

    # 執行偵測
    def doDetect(self, image):
        # 處理將來來源影像灰階化
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 偵測臉部
        faces = self.__faceDetectClass.detectMultiScale(imgGray, scaleFactor=1.08, minNeighbors=5, minSize=(32, 32))
        # 回傳臉部標示四角形座標位置[(x,y,w,h)]
        return faces
