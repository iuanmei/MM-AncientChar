import os
import shutil

# 原始文件夹路径
source_dir = "小学堂"
# 新文件夹路径
target_dir = "小学堂字形"

# 创建新文件夹
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

# 获取一级文件夹列表
subfolders_level_1 = os.listdir(source_dir)

# 遍历一级文件夹
for subfolder_level_1 in subfolders_level_1:
    subfolder_level_1_path = os.path.join(source_dir, subfolder_level_1)
    
    # 确保是文件夹
    if os.path.isdir(subfolder_level_1_path):
        
        # 获取三级文件夹列表
        subfolders_level_3 = os.listdir(subfolder_level_1_path)
        
        # 遍历三级文件夹
        for subfolder_level_3 in subfolders_level_3:
            subfolder_level_3_path = os.path.join(subfolder_level_1_path, subfolder_level_3)
            
            # 确保是文件夹
            if os.path.isdir(subfolder_level_3_path):
                
                # 创建目标文件夹路径
                target_subfolder_path = os.path.join(target_dir, subfolder_level_3)
                
                # 创建目标字形子文件夹路径
                target_shape_folder_path = os.path.join(target_subfolder_path, subfolder_level_1)
                
                # 如果目标字形子文件夹不存在，则创建它
                if not os.path.exists(target_shape_folder_path):
                    os.makedirs(target_shape_folder_path)
                
                # 获取三级文件夹下的图片文件列表
                image_files = os.listdir(subfolder_level_3_path)
                
                # 遍历图片文件，将其复制到目标位置
                for image_file in image_files:
                    image_file_path = os.path.join(subfolder_level_3_path, image_file)
                    target_image_path = os.path.join(target_shape_folder_path, image_file)
                    shutil.copy(image_file_path, target_image_path)

print("数据预处理完成！")

# import os
# import shutil

# # 定义要搜索的根文件夹路径
# root_folder = "小学堂"

# # 遍历根文件夹及其子文件夹
# for root, dirs, files in os.walk(root_folder):
#     for dir_name in dirs:
#         if dir_name == ".ipynb_checkpoints":
#             dir_path = os.path.join(root, dir_name)
#             # 删除文件夹及其内容
#             try:
#                 shutil.rmtree(dir_path)
#                 print(f"已删除文件夹: {dir_path}")
#             except OSError as e:
#                 print(f"删除文件夹时出错: {dir_path} ({e})")


