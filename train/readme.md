# Training

This directory provides a training code for Stable Cascade, as well as guides to download the models you need.
Specifically, you can find training scripts for the following use-cases:
- Text-to-Image
- Image Reconstruction
- ControlNet
- LoRA

#### Note:
A quick clarification, Stable Cascade uses Stage A & B to compress images and Stage C is used for the text-conditional
learning. Therefore, it makes sense to train a LoRA or ControlNet **only** for Stage C. You also don't train a LoRA or 
ControlNet for the Stable Diffusion VAE right?