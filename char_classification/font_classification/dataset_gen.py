import common_util
import os

image_dir = "/home/anyang/projects/classification/img_datas"
files = common_util.file_walk(image_dir)
datas=[]
for f in files:
    if f.endswith("png"):
        file_name=f.lstrip(image_dir)
        bname = os.path.basename(f)
        print(bname)
        label=0
        if "JiaguChar" in f:
            label=0
        elif "JinChar" in f:
            label=1
        elif "XiaozhuanChar" in f:
            label=2
        datas.append({
            "file_name":file_name,
            "labels":label
        })
        # break
    
common_util.save_huggface_json_datas(f"{image_dir}/metadata.jsonl",datas)
