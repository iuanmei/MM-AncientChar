from datasets import load_dataset
from datasets import Dataset, Features,Image, Value,Sequence
import pandas as pd

from PIL import Image as PI
import PIL,os
import json
import argparse

# dir = "./"
# dir = "/Users/fanghui/Downloads/名画合集/001：浮世绘花卉【143P】"
dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"
# dir = "/mnt/task_runtime/data/ancient_char_data/小学堂字形prep"

huggingfacehub_dir = "/mnt/task_runtime/data/fonts_en"

TAG  = "ancient_chars"


# images =[]
# texts = []

input_image=[]
edit_prompt=[]
edited_image=[]
num_bytes = 0

# filename_set = {}



def scan_photo(dir):
    # metadata_path = os.path.join(dir, 'metadata.jsonl')
    metadata_path = os.path.join(dir, 'metadata_en.jsonl')
    # metadata_path = os.path.join(dir, 'metadata_sample.jsonl')

    with open(metadata_path, 'r') as f:
        for line in f:
        # 解析 JSON 对象
            mtdata = json.loads(line)

            in_path = mtdata["input_image"]
            print(in_path)
            ed_path = mtdata["edited_image"]
            # print(ed_path)
            edit = mtdata["edit_prompt"]
            # print(edit)
            
            # in_path = mtdata["img_1"]
            # # print(in_path)
            # ed_path = mtdata["img_2"]
            # edit = mtdata["edit_prompt"]

            try:
                in_image = PI.open(in_path)
                print(in_image)
                ed_image = PI.open(ed_path)

                if hasattr(in_image,"filename"):
                    in_image.filename = ""

                if hasattr(ed_image,"filename"):
                    ed_image.filename = ""
                
                in_bytes = in_image.tobytes()
                ed_bytes = in_image.tobytes()
                num_bytes = len(in_bytes) + len(ed_bytes)
                    # if len(bytes) < 50*1024:
                # if len(bytes) < 30*1024 or len(ed_bytes) < 30*1024:
                #     continue
            
                image_decode = Image(decode=True, id=None)
                in_image = image_decode.encode_example(value=in_image)
                ed_image = image_decode.encode_example(value=ed_image)

                input_image.append(in_image)
                edited_image.append(ed_image)
                edit_prompt.append(edit)
            except:
                print(f"\n")

def save_huggingfacehub_dir(image_dir, huggingfacehub_dir, tag="", hub_token=None):
    global TAG
    TAG = tag
    scan_photo(image_dir)
    # we need to define the features ourselves
    features = Features({
        # 'text': Value(dtype='string', id=None),
        'edit_prompt': Value(dtype='string', id=None),
        # 'image': Image(decode=True, id=None),
        'input_image': Image(decode=True, id=None),
        'edited_image': Image(decode=True, id=None),

    })

    df = pd.DataFrame({
        "edit_prompt": edit_prompt,
        "input_image": input_image,
        "edited_image": edited_image,
        })

    if len(input_image) == len(edit_prompt) == len(edited_image):
        num_examples = len(edit_prompt)
        print(f"total: {num_examples}")
    else:
        print(f"wrong length of triplets: {len(input_image)},{len(edit_prompt)}, {len(edited_image)}")
        # tt = pa.Table.from_pandas(df)
    save_dir = huggingfacehub_dir+"/train"
    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)
    parquet_file = save_dir + "/train.parquet"
    print(f"save {parquet_file}")
    df.to_parquet(parquet_file)
    dataset = Dataset.from_pandas(df, features=features)
    readme = """---
dataset_info:
features:

- name: input_image
    dtype: image
- name: edit_prompt
    dtype: string
- name: edited_image
    dtype: image

splits:
- name: train
- num_bytes: $num_bytes
- num_examples: $count
---
[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
"""
    
    readme = readme.replace("$count", str(num_examples))
    readme = readme.replace("$num_bytes", str(num_bytes))
    readme_file = huggingfacehub_dir+"/README.md"
    if os.path.exists(readme_file):
        os.remove(readme_file)
    with open(readme_file,'a') as f:
        f.write(readme)
        f.close()
        print(f"write readme file {readme_file}")
    # print(f"{tt}")


    if hub_token!=None:
        dataset_name = huggingfacehub_dir[huggingfacehub_dir.rfind("/") + 1:]
        dataset.push_to_hub(repo_id="yuanmei424/fonts_en", split="train", branch="main", token="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", private=False)
        # dataset.push_to_hub(repo_id="xfh/"+dataset_name, split="train", branch="main", token="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", private=True)
    print(f"success!!!!")
    print(f"cd {huggingfacehub_dir} && git add . && git commit -m \"init commit\" && git push")


# """
# python generate_huggingface_dataset.py --image_dir "/Users/fanghui/Downloads/miyutaeokiba" \
# --huggingfacehub_dir "/Users/fanghui/dev/miyutaeokiba" \
# --tag "miyutaeokiba"
# """

if __name__ == "__main__":
    image_dir = "/mnt/task_runtime/data/ancient_char_data/历代字形prep"
    huggingfacehub_dir = "/mnt/task_runtime/data/fonts_en"
    TAG  = "ancient_chars"
    parser = argparse.ArgumentParser(description="Simple example of generate dataset and upload huggingfacehub  script.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        required=True,
        help="image dir, as: /tmp/images",
    )
    parser.add_argument(
        "--huggingfacehub_dir",
        type=str,
        default=None,
        required=True,
        help="huggingfacehub dataset dir is project name",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        required=False,
        help="file tag name",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default="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",
        required=False,
        help="huggingfacehub token",
    )
    args = parser.parse_args()

    save_huggingfacehub_dir(args.image_dir, args.huggingfacehub_dir, args.tag, args.hub_token)

# python /mnt/task_runtime/prep/ds_tohub.py --image_dir /mnt/task_runtime/data/ancient_char_data/历代字形prep --huggingfacehub_dir /mnt/task_runtime/data/fonts_en

