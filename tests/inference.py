import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline


model_id = "yuanmei424/xxt_en_instructpix2pix"
# model_id = "/mnt/task_runtime/diffusers/examples/instruct_pix2pix/instruct-pix2pix-model"
# model_id = "yuanmei424/xxt_instructpix2pix"  # <- replace this
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None,
    requires_safety_checker = False).to("cuda")
# pipe.safety_checker = lambda images, clip_input: (images, False)
generator = torch.Generator("cuda").manual_seed(1)

# url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"

# image_path = "/mnt/task_runtime/data/ancient_char_data/历代字形/鸟/甲骨文/5.jpg"
# # prompt = "汉字[鸟]从甲骨文演化为金文"
# prompt = "The Chinese character is transformed from oracle bone inscription to clerical script."
# save_path = "/mnt/task_runtime/tests/鸟/fonts_en.png"



# image_path = "/mnt/task_runtime/tests/水/甲骨/4.png"
# prompt = "The Chinese character is transformed from oracle bone inscription to bronze ware script."
# save_path = "/mnt/task_runtime/tests/水/fonts_en.png"



# image_path = "/mnt/task_runtime/tests/一/简帛/1.png"
# image_path ="/mnt/task_runtime/tests/一/甲骨/1.png"


image_path = "/mnt/task_runtime/tests/妻/简帛/10.png"
# prompt = "汉字[妻]从简帛演化为小篆"
prompt = "The Chinese character 妻 is transformed from silk script to seal script. "
# prompt = "The Chinese character 一 is transformed from oracle bone inscription to bronze ware script. "
save_path = "/mnt/task_runtime/tests/妻/fonts_en.png"








def pix2pix(image_path, prompt,save_path):
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
   # num_inference_steps = 200
   #不敏感

   # image_guidance_scale = 1.5
   # image_guidance_scale = 1.0
   image_guidance_scale = 0.9
   # image_guidance_scale = 0.05
   ## >1 变彩色越大越彩  <1都一样

   # guidance_scale = 1
   guidance_scale = 20
   # 不敏感


   edited_image = pipe(
      prompt,
      image=image,
      num_inference_steps=num_inference_steps,
      image_guidance_scale=image_guidance_scale,
      guidance_scale=guidance_scale,
      generator=generator,
   ).images[0]
   edited_image.save(save_path)

pix2pix(image_path, prompt,save_path)