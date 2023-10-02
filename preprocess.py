import cv2
from PIL import Image
import numpy as np

def is_black_background(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的平均像素值
    average_pixel_value = np.mean(gray)

    # 根据平均像素值判断背景颜色
    if average_pixel_value < 128:
        return True
    else:
        return False

def convert_to_white_background(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值化处理，将黑色文字变为白色
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 创建一个白色背景
    white_background = np.ones_like(img) * 255

    # 保留黑色文字部分
    result = cv2.bitwise_and(white_background, white_background, mask=sure_bg)

    # 保存处理后的图像
    cv2.imwrite("output_image.png", result)

if __name__ == "__main__":
    # image_path = "input_image.png"
    image_path = "/Users/april/Desktop/chars/B00178B00178_2_270.png"

    if is_black_background(image_path):
        convert_to_white_background(image_path)
        print("已转换为白底黑字的图片并去噪")
    else:
        print("图片已经是白底黑字的，无需转换")
