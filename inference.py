import os
import torch
import glob
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import json
import argparse

from fiva_adapter import FivaAdapterXL


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
qformer_pretrained_path = "salesforce/blipdiffusion"
ip_ckpt = "fiva_models/ip_adapter.bin"
qformer_ckpt = "fiva_models/pytorch_model_1.bin"
device = "cuda"
max_ref_attrs = 3

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# load SD pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
ip_model = FivaAdapterXL(pipe, qformer_pretrained_path, ip_ckpt, qformer_ckpt, device, max_ref_attrs=max_ref_attrs)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=203,
    )
    args = parser.parse_args()
    return args


data_dir = './validation/full_imgs/'
meta_file = './validation/full_meta.json'
output_dir = './results/output_images/'

os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    args = parse_args()

    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    for meta in meta_data[args.start:args.end]:
        prompts = []
        tags = []
        src_prompt = meta['target_subjects']['itself']
        image_path = os.path.join(data_dir, meta['image_file'])
        img_id = meta['image_file'].strip().split('.png')[0]
        for k, vs in meta['target_subjects'].items():
            for v in vs:
                tags.append(k)
                prompts.append(v)
        idx = int(img_id)
        attr_type = meta['attribute']
        for prompt, tag, in zip(prompts, tags):
            file_id = image_path.split('/')[-1].split('.')[0]

            # read image prompt
            image = Image.open(image_path)
            image = image.resize((512, 512))

            # generate image variations
            image_out = ip_model.generate(pil_image=[image], attr_text=[attr_type], num_samples=1, num_inference_steps=30, seed=420, prompt=[prompt], size=(512,512), scale=0.6)[0]
            tag = tag.replace('/', '_')
            prompt = prompt.replace('/', '_')
            image_file = '{}/{}_{}_{}_{}.png'.format(output_dir, img_id, attr_type, tag, prompt)
            image_file = image_file.replace(' ', '_')
            image_out.save(image_file)

