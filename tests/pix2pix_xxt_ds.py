import os
import json


import PIL
import requests
import torch

from pathlib import Path

from diffusers import StableDiffusionInstructPix2PixPipeline




# model_id = "yuanmei424/xxt_instructpix2pix"  # <- replace this
# model_id = "yuanmei424/xxt_en_instructpix2pix"  # <- replace this
model_id = "yuanmei424/fonts_en_instructpix2pix"


pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None,
    requires_safety_checker = False).to("cuda")
# pipe.safety_checker = lambda images, clip_input: (images, False)
generator = torch.Generator("cuda").manual_seed(1)

# url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"

# url = "/mnt/task_runtime/data/ancient_char_data/历代字形/鸟/甲骨文/5.jpg"


def pix2pix(image_path, prompt, save_path):
   # image_path = "/mnt/task_runtime/tests/妻/金文/0.png"

   def download_image(image_path):
   #    image = PIL.Image.open(requests.get(url, stream=True).raw)
      image = PIL.Image.open(image_path)
      image = PIL.ImageOps.exif_transpose(image)
      image = image.convert("RGB")
      return image

   image = download_image(image_path)
   # prompt = "汉字[鸟]从甲骨文演化为简帛"
   # prompt = "汉字[妻]从金文演化为小篆"

   num_inference_steps = 20
   # image_guidance_scale = 1.5
   image_guidance_scale = 0.9
   # guidance_scale = 10
   guidance_scale = 20


   edited_image = pipe(
      prompt,
      image=image,
      num_inference_steps=num_inference_steps,
      image_guidance_scale=image_guidance_scale,
      guidance_scale=guidance_scale,
      generator=generator,
   ).images[0]
   edited_image.save(save_path)


# metadata_path = "/mnt/task_runtime/tests/test.jsonl"
# metadata_path = "/mnt/task_runtime/tests/test_sample.jsonl"
metadata_path = "/mnt/task_runtime/tests/test_sample_en.jsonl"


# metadata_path = os.path.join(dir, 'metadata_sample.jsonl')

with open(metadata_path, 'r') as f:
    for line in f:
    # 解析 JSON 对象
        mtdata = json.loads(line)

      #   image_path = mtdata["image_path"]
        image_path = mtdata["input_image"]
        # print(in_path)
        save_path = mtdata["save_path"]
      #   prompt = mtdata["prompt"]
        prompt = mtdata["edit_prompt"]
        
        bname = os.path.basename(save_path)
      #   save_path = save_path.replace("xxt_sample_res","xxt_en_res")
        save_path = save_path.replace("xxt_sample_res","fonts_en_res")

        out_path = Path(save_path.rstrip(bname))
        print("out_path",out_path)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        
        pix2pix(image_path, prompt, save_path)


# nohup python /mnt/task_runtime/tests/pix2pix_xxt_ds.py > /mnt/task_runtime/logs/xxt_ds_test.log 2>&1 &