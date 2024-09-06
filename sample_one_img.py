from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image


model_path = "/ckp/path/"
image = Image.open("demos/demo_img.jpg")
mask_image = Image.open("demos/demo_mask.jpg")

prompt = 'Polyp'

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe = pipe.to("cuda")

gen_image = pipe(prompt=prompt, image=image, mask_image=mask_image,
    width=image.size[0], height=image.size[1], num_inference_steps=50,
        ).images[0]

gen_image.save("sample.jpg")


