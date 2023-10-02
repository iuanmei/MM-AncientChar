import json
import os

# mtdata_dir = "/mnt/task_runtime/data/ancient_char_data/小学堂字形prep"
mtdata_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"


# testjson = "/mnt/task_runtime/tests/test_sample.jsonl"
# 打开metadata.jsonl文件
# with open(testjson, "r", encoding="utf-8") as input_file:

with open( os.path.join(mtdata_dir, "metadata_sample.jsonl"), "r", encoding="utf-8") as input_file:
# with open( os.path.join(mtdata_dir, "metadata.jsonl"), "r", encoding="utf-8") as input_file:
    lines = input_file.readlines()

# 初始化用于存储处理后数据的列表
edited_data = []

# 定义替换规则
replacement_rules = {
    "汉字": "The Chinese character ",
    "简帛": " silk script",
    "小篆": " seal script",
    "甲骨": " oracle bone inscription",
    "金文": " bronze ware script",
    "从": " is transformed from",
    "演化为": " to",
    "from战国文字": "from Warring States script",
    "inscription文": "inscription",
    "to隶书": "to clerical script",
    "to秦文字": "to Qin script"
}

# 处理每一行数据
for line in lines:
    data = json.loads(line.strip())  # 解析JSON数据
    
    # 替换"edit_prompt"的值
    edit_prompt = data["edit_prompt"]
    for key, value in replacement_rules.items():
        edit_prompt = edit_prompt.replace(key, value)
    
    edit_prompt = edit_prompt.replace("[", "").replace("]", "")
    # 更新"edit_prompt"的值
    data["edit_prompt"] = edit_prompt
    
    edited_data.append(data)

# 将处理后的数据写入metadata_en.jsonl文件，指定编码为UTF-8
# with open("/mnt/task_runtime/tests/test_sample_en.jsonl", "w", encoding="utf-8") as output_file:
with open(os.path.join(mtdata_dir,"metadata_sample_en.jsonl"), "w", encoding="utf-8") as output_file:
# with open(os.path.join(mtdata_dir,"metadata_en.jsonl"), "w", encoding="utf-8") as output_file:
    for item in edited_data:
        output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
