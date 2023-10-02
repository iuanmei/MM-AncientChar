import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"/mnt/task_runtime/prep/觥/小篆/0.png", -1)
print(image.shape)
# 交换第三和第四通道数据

if image.shape[2] == 4:
    image[:,:,1] = image[:,:,3]

    # 删除第四通道数据
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    img=image
    # 打印图像的维度
    print("Image shape:", img.shape)

img=image

# 图像BGR转RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 图像灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行二值化操作，将灰度图像中小于某个值的设为0，大于某个值的设为255，这里设定阈值为127
# 你可以根据实际提供的图片，尝试调整这个阈值，以更好地将背景和对象区分开
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 寻找对象的轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个全白图片
white = np.ones_like(img_rgb) * 255

# 将对象的轮廓画在全白图片上
image = cv2.drawContours(white, contours, -1, (0, 0, 0), 2)

cv2.imwrite("/mnt/task_runtime/prep/觥/小篆/000p.png", image)