# import os
# import random
# import json

# # 获取所有图片的路径
# def get_all_images(root_path):
#     images = {}
#     for char in os.listdir(root_path):
#         char_path = os.path.join(root_path, char)
#         if os.path.isdir(char_path):
#             images[char] = {}
#             for style in os.listdir(char_path):
#                 style_path = os.path.join(char_path, style)
#                 if os.path.isdir(style_path):
#                     images[char][style] = [os.path.join(style_path, img) for img in os.listdir(style_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
#     return images

# # 生成metadata
# def generate_metadata(images):
#     metadata = []
#     for char, styles in images.items():
#         if len(styles) >= 2:
#             style_pairs = list(styles.keys())
#             random.shuffle(style_pairs)
#             for i in range(len(style_pairs) - 1):
#                 for j in range(i + 1, len(style_pairs)):
#                     style1 = style_pairs[i]
#                     style2 = style_pairs[j]
#                     img1 = random.choice(images[char][style1])
#                     img2 = random.choice(images[char][style2])
#                     entry = {
#                         "img_1": img1,
#                         "edit_prompt": f"汉字[{char}]从{style1}演化为{style2}",
#                         "img_2": img2
#                     }
#                     metadata.append(entry)
#     return metadata

# # 主函数
# def main():
#     root_path = "历代字形"
#     images = get_all_images(root_path)
#     metadata = generate_metadata(images)
    
#     with open(os.path.join(root_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
#         for entry in metadata:
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# if __name__ == "__main__":
#     main()



import os
import json

# 定义文件夹路径
# base_dir = "历代字形"
base_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"


# 获取所有二级子文件夹（汉字字符）
characters = [char for char in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, char))]

# 创建一个空列表来存储所有的metadata
metadata_list = []

# 遍历每一个汉字字符
for char in characters:
    char_dir = os.path.join(base_dir, char)
    
    # 获取该汉字字符下的所有三级子文件夹（字形）
    forms = [form for form in os.listdir(char_dir) if os.path.isdir(os.path.join(char_dir, form))]
    
    # 对每一个字形进行两两组合
    for i in range(len(forms)):
        for j in range(i+1, len(forms)):
            form1_dir = os.path.join(char_dir, forms[i])
            form2_dir = os.path.join(char_dir, forms[j])
            
            # 获取两个字形下的所有图片
            form1_imgs = [img for img in os.listdir(form1_dir) if img.endswith(('.jpg','jpeg','.png'))]
            form2_imgs = [img for img in os.listdir(form2_dir) if img.endswith(('.jpg','jpeg','.png'))]
            
            # 遍历两个字形下的所有图片组合
            for img1 in form1_imgs:
                for img2 in form2_imgs:
                    img1_path = os.path.join(form1_dir, img1)
                    img2_path = os.path.join(form2_dir, img2)
                    
                    # 创建metadata
                    metadata = {
                        "input_image": img1_path,
                        "edit_prompt": f"汉字[{char}]从{forms[i]}演化为{forms[j]}",
                        "edited_image": img2_path
                    }
                    
                    metadata_list.append(metadata)

# 将metadata_list写入metadata.jsonl文件
with open(os.path.join(base_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for item in metadata_list:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
