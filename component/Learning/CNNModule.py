from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense


# 卷積神經網路模型物件
class CNNModule():

    labels = ["Bob", "None"]
    __model = None

    # 初始化
    def __init__(self):
        # 建立線性堆疊模型容器
        model = Sequential()
        # 一個卷積計算，包含一個卷積層與一個池化層
        # 加入第一個卷積層
        model.add(
            Conv2D(
                filters=32,  # 隨機產生32個filter
                kernel_size=(3, 3),  # 每一個filter size為3*3
                padding="same",  # 卷積運算中，產生的卷積影像大小不變
                input_shape=(32, 32, 3),  # 指定輸入的影像是32*32且為RGB的3個色度
                activation="relu"  # 設定為ReLU運作函式
            )
        )
        # 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
        model.add(
            Dropout(rate=0.25)
        )
        # 加入第一個池化層，這層用意在縮減取樣數，把32*32縮減為16*16，但數量維持不變為32個樣本
        model.add(
            MaxPooling2D(pool_size=(2, 2))
        )
        # 加入第二個卷積層
        model.add(
            Conv2D(
                filters=64,  # 隨機產生64個filter
                kernel_size=(3, 3),  # 每一個filter size為3*3
                padding="same",  # 卷積運算中，產生的卷積影像大小不變
                activation="relu"  # 設定為ReLU運作函式
            )
        )
        # 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
        model.add(
            Dropout(rate=0.25)
        )
        # 加入第二個池化層，這次是第二次縮減取樣，所以會把16*16縮減為8*8，但數量維持為64個樣本
        model.add(
            MaxPooling2D(pool_size=(2, 2))
        )

        #####建立神經網路#####
        # 加入平坦層
        # 將池化過64個8*8影像影像轉為一層陣列，64*8*8=4096，會有4096個float數字，對應到了4096個神經元
        model.add(
            Flatten()
        )
        # 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
        model.add(
            Dropout(rate=0.25)
        )
        # 加入隱藏層
        # 到了這層，神經元會變成1024個，並且加上非線性激活函式，避免訓練結果變成線性，這邊一樣會隨機放棄25%的神經元
        model.add(
            Dense(1024, activation='relu')
        )
        # 加入一個Dropout層，在訓練迭代中，隨機放棄神經元，避免overfitting，Dropout(0.25)表示隨機放棄25%的神經元
        model.add(
            Dropout(rate=0.25)
        )
        # 加入輸出層，最後的結果我們分成10個影像類別(10個神經元)，並使用激活函式softmax輸出結果，轉換成預測每個類別的機率
        model.add(
            Dense(2, activation='softmax')
        )
        # 輸出外界供使用
        self.__model = model

    # 取得模型
    def getModel(self):
        return self.__model
