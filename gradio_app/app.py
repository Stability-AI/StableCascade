import os
import random
import gradio as gr
import numpy as np
import PIL.Image
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DESCRIPTION = "# Stable Cascade"
DESCRIPTION += "\n<p style=\"text-align: center\">Unofficial demo for <a href='https://huggingface.co/stabilityai/stable-cascade' target='_blank'>Stable Casacade</a>, a new high resolution text-to-image model by Stability AI, built on the Würstchen architecture - <a href='https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE' target='_blank'>non-commercial research license</a></p>"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") != "0"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1536"))
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"

device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print("RUNNING ON:", device)

c_dtype = torch.bfloat16 if device.type == "cpu" else torch.float
prior_pipeline = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=c_dtype)#.to(device)
decoder_pipeline = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.half)#.to(device) 

if ENABLE_CPU_OFFLOAD:
    prior_pipeline.enable_model_cpu_offload()
    decoder_pipeline.enable_model_cpu_offload()
else:
    prior_pipeline.to(device)
    decoder_pipeline.to(device)

if USE_TORCH_COMPILE:
    prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
    decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="max-autotune", fullgraph=True)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@torch.inference_mode()
def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 20,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 10,
    decoder_guidance_scale: float = 0.0,
    num_images_per_prompt: int = 2,
) -> PIL.Image.Image:
    prior_pipeline.to(device)
    decoder_pipeline.to(device)

    generator = torch.Generator().manual_seed(seed)
    print("prior_num_inference_steps: ", prior_num_inference_steps)
    prior_output = prior_pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=prior_num_inference_steps,
        timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        negative_prompt=negative_prompt,
        guidance_scale=prior_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )

    decoder_output = decoder_pipeline(
        image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        num_inference_steps=decoder_num_inference_steps,
        guidance_scale=decoder_guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="pil",
    ).images

    return decoder_output[0]


examples = [
    "An astronaut riding a green horse",
    "A mecha robot in a favela by Tarsila do Amaral",
    "The sprirt of a Tamagotchi wandering in the city of Los Angeles",
    "A delicious feijoada ramen dish"
]

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
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
        with gr.Row():
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
                value=1,
            )
        with gr.Row():
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

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    inputs = [
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            prior_num_inference_steps,
            prior_guidance_scale,
            decoder_num_inference_steps,
            decoder_guidance_scale,
            num_images_per_prompt,
    ]
    gr.on(
        triggers=[prompt.submit, negative_prompt.submit, run_button.click],
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )

with gr.Blocks(css="gradio_app/style.css") as demo_with_history:
    with gr.Tab("App"):
        demo.render()

if __name__ == "__main__":
    demo_with_history.launch()
