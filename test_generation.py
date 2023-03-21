import torch

from pathlib import Path

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline


def simple_image():
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to("cuda")
    pipeline("An image of a squirrel in Picasso style").images[0]


def pndm_image():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # prompt = "a photo of an astronaut riding a horse on mars"
    # prompt = "a cat covered in sunflowers"

    save_path = Path("images")

    prompt = "bank robbery using bananas"
    # prompt = "classical painting of an united nations assembly of crickets"
    name = prompt.replace(" ", "_")
    n_smampes = 10
    for ii in range(n_smampes):
        image = pipe(prompt).images[0]
        filename = f"{name}_{ii}.png"
        image.save(save_path / filename)


if (__name__) == "__main__":
    # simple_image()
    pndm_image()
