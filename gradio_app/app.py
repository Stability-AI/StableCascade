#@title Load models
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print("RUNNING ON:", device)

c_dtype = torch.bfloat16 if device.type == "cpu" else torch.float
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=c_dtype)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.half)
prior.to(device)
decoder.to(device)

import random
import gc
import numpy as np
import gradio as gr

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1536

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def generate_prior(prompt, negative_prompt, generator, width, height, num_inference_steps, guidance_scale, num_images_per_prompt):
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps
    )
    torch.cuda.empty_cache()
    gc.collect()
    return prior_output.image_embeddings


def generate_decoder(prior_embeds, prompt, negative_prompt, generator, num_inference_steps, guidance_scale):
    decoder_output = decoder(
        image_embeddings=prior_embeds.to(device=device, dtype=decoder.dtype),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images
    torch.cuda.empty_cache()
    gc.collect()
    return decoder_output


@torch.inference_mode()
def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    randomize_seed: bool = True,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 20,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 10,
    decoder_guidance_scale: float = 0.0,
    num_images_per_prompt: int = 2,
):
    """Generate images using Stable Cascade."""
    seed = randomize_seed_fn(seed, randomize_seed)
    print("seed:", seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    prior_embeds = generate_prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        width=width,
        height=height,
        num_inference_steps=prior_num_inference_steps,
        guidance_scale=prior_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,

    )

    decoder_output = generate_decoder(
        prior_embeds=prior_embeds,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=decoder_num_inference_steps,
        guidance_scale=decoder_guidance_scale,
    )

    return decoder_output


examples = [
    "An astronaut riding a green horse",
    "A mecha robot in a favela by Tarsila do Amaral",
    "The sprirt of a Tamagotchi wandering in the city of Los Angeles",
    "A delicious feijoada ramen dish"
]

with gr.Blocks(css="gradio_app/style.css") as demo:
    with gr.Column():
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            placeholder="Enter your prompt",
        )
        run_button = gr.Button("Run")
        with gr.Accordion("Advanced options", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a Negative Prompt",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            width = gr.Slider(
                label="Width",
                minimum=1024,
                maximum=MAX_IMAGE_SIZE,
                step=128,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=1024,
                maximum=MAX_IMAGE_SIZE,
                step=128,
                value=1024,
            )
            num_images_per_prompt = gr.Slider(
                label="Number of Images",
                minimum=1,
                maximum=2,
                step=1,
                value=2,
            )
            prior_guidance_scale = gr.Slider(
                label="Prior Guidance Scale",
                minimum=0,
                maximum=20,
                step=0.1,
                value=4.0,
            )
            prior_num_inference_steps = gr.Slider(
                label="Prior Inference Steps",
                minimum=10,
                maximum=30,
                step=1,
                value=20,
            )

            decoder_guidance_scale = gr.Slider(
                label="Decoder Guidance Scale",
                minimum=0,
                maximum=0,
                step=0.1,
                value=0.0,
            )
            decoder_num_inference_steps = gr.Slider(
                label="Decoder Inference Steps",
                minimum=4,
                maximum=12,
                step=1,
                value=10,
            )
    with gr.Column():
        result = gr.Gallery(label="Result", show_label=False)

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
    )

    inputs = [
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            prior_num_inference_steps,
            prior_guidance_scale,
            decoder_num_inference_steps,
            decoder_guidance_scale,
            num_images_per_prompt,
    ]
    prompt.submit(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    negative_prompt.submit(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    run_button.click(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )

demo.queue(20).launch()
