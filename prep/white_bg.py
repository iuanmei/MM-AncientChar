import cv2
from PIL import Image
import numpy as np
import os
import common_util
from pathlib import Path



## 1. 读取输入图片并处理通道
def process_image(input_image_path):
    # 读取图片
    image = cv2.imread(input_image_path,-1)
    print(image.shape)
    # print(image.tolist())
    if image.shape[-1] == 4:
        mask = image[:, :, 3] == 0 
        image[mask,:3] = [255,255,255]
        image[:,:,0] = image[:,:,3]
        # print(image.tolist())
        # 删除第四通道数据
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    elif image.shape[-1] != 3:
        # print(image.shape)
        raise ValueError("Input image should have 3 or 4 channels.")
    
    return image







# 2. 判断背景颜色并颜色反转
def convert_to_white_background(image):
    # 转换为灰度图像
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[gray_image <= 200] = [255, 255, 255]
    
    # _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    
    # 计算图像的平均亮度
    mean_brightness = np.mean(gray_image)
    # print("mean_brightness",mean_brightness)
    # 判断背景颜色是白色还是黑灰色
    if mean_brightness > 128:
        # 图像背景是白色，确保线条轮廓是黑色
        inverted_image = cv2.bitwise_not(image)
        
    else:
        # 图像背景是黑灰色，转换为白色背景黑色线条
        
        # inverted_image = cv2.bitwise_not(gray_image)
        inverted_image = cv2.cvtColor(cv2.bitwise_not(gray_image), cv2.COLOR_GRAY2BGR)
        # inverted_image = cv2.bitwise_not(thresholded_image)
        # return image
    
    return inverted_image

# # 3. 去噪和增强对比度
def enhance_image_gaussian(image):
    # 使用高斯模糊去噪
    denoised_image = cv2.GaussianBlur(image, (3, 3), 1.0)

    # 中值滤波
    # denoised_image = cv2.medianBlur(image, 5)

    # 增强对比度
    gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    return enhanced_image

    # ##中值滤波
    # denoised_image = cv2.medianBlur(image, 5)
    # gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    # denoised_contrast_image = cv2.equalizeHist(gray)
    
    # return denoised_contrast_image

# # 3. 去噪和增强对比度
def enhance_image_midblur(image):
    # 使用高斯模糊去噪
    # denoised_image = cv2.GaussianBlur(image, (3, 3), 1.0)

    # 中值滤波
    denoised_image = cv2.medianBlur(image, 5)

    # denoised_image = cv2.medianBlur(image, 5)  # 第一次中值滤波
    # denoised_image = cv2.medianBlur(denoised_image, 5)  # 第二次中值滤波


    # 增强对比度
    gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    return enhanced_image


# 3. 去噪和增强对比度
# def enhance_image(image):
#     # 使用高斯模糊去噪
#     blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
#     # ##中值滤波
#     # denoised_image = cv2.medianBlur(image, 5)

#     # 分离通道并分别增强对比度
#     b, g, r = cv2.split(blurred_image)
#     enhanced_b = cv2.equalizeHist(b)
#     enhanced_g = cv2.equalizeHist(g)
#     enhanced_r = cv2.equalizeHist(r)

#     # 合并增强后的通道
#     enhanced_image = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
    
#     return enhanced_image

# 4. 存储处理后的图片
def save_image(output_image_path, image):
    cv2.imwrite(output_image_path, image)

def main(input_image_path,output_image_path):
    # 读取图片
    image= cv2.imread(input_image_path,-1)
    flag = 1 if image.shape[-1] != 3 else 0
    # 处理图片
    processed_image = process_image(input_image_path)
    # print(processed_image.shape)
    inverted_image = convert_to_white_background(processed_image)
    # print(inverted_image.shape)
    
    enhanced_image = enhance_image_gaussian(inverted_image) if flag ==1 else enhance_image_midblur(inverted_image)

    # print(enhanced_image.shape)
    # 存储处理后的图片
    save_image(output_image_path, enhanced_image)


if __name__ == "__main__":
    dirs = 0

    
    if dirs == 0:
        input_image_path = "/mnt/task_runtime/prep/牧/金文/2.jpg"  # 替换为您的输入图片路径
        output_image_path = "/mnt/task_runtime/prep/牧/金文/2p.jpg"     # 替换为您的输出图片路径

        main(input_image_path,output_image_path)

    else:
    # image_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形"
    # output_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"
        image_dir = "/mnt/task_runtime/data/ancient_char_data/小学堂字形"
        output_dir = "/mnt/task_runtime/data/ancient_char_data/小学堂字形prep"
        files = common_util.file_walk(image_dir)
        
        for f in files:
            if f.endswith(('jpg','png','jpeg')):
                # file_name=f.rstrip(image_dir)
                # print("file_name1=",file_name)
                file_name=f.lstrip(image_dir)
                # print("processed",file_name)
                # bname = os.path.basename(f)
                # print("bname=",bname)
                output_file = f"{output_dir}/{file_name}"
                # print("output_file",output_file)
                bname = os.path.basename(output_file)
                out_path = Path(output_file.rstrip(bname))
                # print("out_path",out_path)
                Path(out_path).mkdir(parents=True, exist_ok=True)
        
                main(f, output_file)
                print("processed",output_file)

        
        

