
from PIL import Image
import os
import common_util
from pathlib import Path
import cv2
import numpy as np
# url = "/mnt/task_runtime/prep/觥/小篆/0.png"



def prep(img_path,out_path):
    image = Image.open(img_path)
    # print(image)
    if image.mode=="RGBA":
        # 转换为RGB
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image,mask=image.split()[3])
        print(rgb_image)
        rgb_image.save(out_path)
    else:
        image = cv2.imread(img_path,-1)
        if image.shape[-1] == 3:
            print(image.shape)
            cv2.imwrite(out_path, image)
        if image.shape[-1] == 2:
            print(image.shape)
            third_channel = np.copy(image[:, :, 0])  # Assuming you want to copy the first channel
            # Stack the three channels together to create a three-channel image
            image_3_channel = cv2.merge([image[:, :, 0], image[:, :, 1], third_channel])
            # Save or use the new three-channel image
            cv2.imwrite(out_path, image_3_channel)
        
    # return rgb_image


image_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形"
output_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"
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

        prep(f, output_file)
        
        print("processed",output_file)
