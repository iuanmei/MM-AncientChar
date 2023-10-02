import os
import json

# 根目录路径
root_dir = "/mnt/task_runtime/data/ancient_char_data/小学堂字形prep_res"

# 初始化用于存储所有数据的列表
data = []

# 遍历一级文件夹
for first_level_dir in os.listdir(root_dir):
    first_level_path = os.path.join(root_dir, first_level_dir)
    # print(first_level_path)
    # 如果是文件夹
    if os.path.isdir(first_level_path):
        
        # 初始化演变信息
        evolution_info = f"汉字[{first_level_dir}]从"
        
        # 遍历二级文件夹
        for second_level_dir in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level_dir)
            
            # 如果包含"甲骨"，"金文"，"简帛"其中之一
            if "甲骨" in second_level_dir:
                evolution_info += "甲骨文演化到金文"
            elif "金文" in second_level_dir:
                evolution_info += "金文演化到简帛"
            elif "简帛" in second_level_dir:
                evolution_info += "简帛演化到小篆"
            
            # 遍历三级文件夹
            for filename in os.listdir(second_level_path):
                file_path = os.path.join(second_level_path, filename)
                
                # 检查如果是文件而不是文件夹
                if os.path.isfile(file_path) and filename.endswith(".png"):
                    # 生成JSONL数据
                    json_data = {
                        "image_path": file_path,
                        "prompt": evolution_info,
                        "save_path": file_path.replace(".png", "_res.png")
                    }
                    
                    data.append(json_data)

# 将数据写入test.jsonl文件
with open("test.jsonl", "w", encoding="utf-8") as jsonl_file:
    for item in data:
        jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")