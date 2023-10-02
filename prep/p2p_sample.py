import os
import random
import json

# 获取所有图片的路径
def get_all_images(root_path):
    images = {}
    for char in os.listdir(root_path):
        char_path = os.path.join(root_path, char)
        if os.path.isdir(char_path):
            images[char] = {}
            for style in os.listdir(char_path):
                style_path = os.path.join(char_path, style)
                if os.path.isdir(style_path):
                    images[char][style] = [os.path.join(style_path, img) for img in os.listdir(style_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    return images

# 生成metadata
def generate_metadata(images):
    metadata = []
    for char, styles in images.items():
        if len(styles) >= 2:
            style_pairs = list(styles.keys())
            random.shuffle(style_pairs)
            for i in range(len(style_pairs) - 1):
                for j in range(i + 1, len(style_pairs)):
                    style1 = style_pairs[i]
                    style2 = style_pairs[j]
                    if len(images[char][style1]) > 0 and len(images[char][style2]) > 0:
                        img1 = random.choice(images[char][style1])
                        img2 = random.choice(images[char][style2])
                        entry = {
                            "img_1": img1,
                            "edit_prompt": f"汉字[{char}]从{style1}演化为{style2}",
                            "img_2": img2
                        }
                        metadata.append(entry)
    return metadata

# 主函数
def main():
    root_path = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"
    images = get_all_images(root_path)
    metadata = generate_metadata(images)
    
    with open(os.path.join(root_path, "metadata_sample.jsonl"), "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()