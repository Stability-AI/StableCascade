# Training
<p align="center">
    <img src="../figures/collage_3.jpg" width="800">
</p>

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

## Basics
In the [training configs](../configs/training) folder we provide config files for all trainings. All config files
follow a similar structure and only contain the most essential parameters you need to set. Let's take a look at the 
structure each config follows:

At first, you will set the run name, checkpoint-, & output-folder and which version you want to train.
```yaml
experiment_id: stage_c_3b_controlnet_base
checkpoint_path: /path/to/checkpoint
output_path: /path/to/output
model_version: 3.6B
```

Next, you can set your [Weights & Biases]() information if you want to use it for logging.
```yaml
wandb_project: StableCascade
wandb_entity: wandb_username
```

Afterwards, you define the training parameters.
```yaml
lr: 1.0e-4
batch_size: 256
image_size: 768
multi_aspect_ratio: [1/1, 1/2, 1/3, 2/3, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6, 9/16]
grad_accum_steps: 1
updates: 500000
backup_every: 50000
save_every: 2000
warmup_updates: 1
use_fsdp: False
```

Most, of them will be quite familiar to you probably already. A few clarification tho: `updates` refers to the number of
training steps, `backup_every` creates additional checkpoints, so you can revert to earlier ones if you want, 
`save_every` concerns how often models will be saved and sampling will be done. Furthermore, since distributed training
is essential when training large models from scratch or doing large finetunes, we have an option to use PyTorch's
[**Fully Shared Data Parallel (FSDP)**](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/). You
can use it by setting `use_fsdp: True`. Note, that you will need multiple GPUs for FSDP. However, this as mentioned 
above, this is only needed for large runs. You can still train and finetune our largest models on a powerful single
machine. <br><br>
Another thing we provide is training with **Multi-Aspect-Ratio**. You can set the aspect ratios you want in the list 
for `multi_aspect_ratio`.<br><br>

For diffusion models, having an EMA (Exponential Moving Average) model, can drastically improve the performance of
your model. To include an EMA model in your training you can set the following parameters, otherwise you can just
leave them away.
```yaml
ema_start_iters: 5000
ema_iters: 100
ema_beta: 0.9
```

Next, you can define the dataset that you want to use. Note, that the code uses 
[webdataset](https://github.com/webdataset/webdataset) for this.
```yaml
webdataset_path:
  - s3://path/to/your/first/dataset/on/s3
  - file:/path/to/your/local/dataset.tar
```
You can set as many dataset paths as you want, and they can either be on 
[Amazon S3 storage](https://aws.amazon.com/s3/) or just local.


## Dataset
As mentioned above, the code uses [webdataset](https://github.com/webdataset/webdataset) for working with datasets,
because this library supports working with large amounts of data very easily. In case you want to **finetune** a model,
train a **LoRA** or train a **ControlNet**, you might not have them in a webdataset format. Therefore, here follows
a simple example how you can convert your dataset into the appropriate format.
1. Put all your images and captions into a folder
2. Rename them to have the same number / id as the name. For example: 
`0000.jpg, 0000.txt, 0001.jpg, 0001.txt, 0002.jpg, 0002.txt, 0003.jpg, 0003.txt`
3. Run the following command: ``tar --sort=name -cf dataset.tar dataset/`` or manually create a tar file from the folder
4. Set the `webdataset_path: file:/path/to/your/local/dataset.tar` in the config file

Next, there are a few more settings that might be helpful to you, especially when working with large datasets that
might contain more information about images, like some kind of variables that you want to filter for. You can apply
dataset filters like the following in the config file:
```yaml
 dataset_filters:
   - ['aesthetic_score', 'lambda s: s > 4.5']
   - ['nsfw_probability', 'lambda s: s < 0.01']
```
In this case, you would have `0000.json, 0001.json, 0002.json, 0003.json` in your dataset as well, with keys for 
`aesthetic_score` and `nsfw_probability`. 

## Text-to-Image Training
## ControlNet Training
## LoRA Training
To train a LoRA on Stage C, you have a few more parameters available to set for the training. 
```yaml
module_filters: ['.attn']
rank: 4
train_tokens:
  # - ['^snail', null] # token starts with "snail" -> "snail" & "snails", don't need to be reinitialized
  - ['[fernando]', '^dog</w>'] # custom token [snail], initialize as avg of snail & snails
```
These include the `module_filters`, which simply determines on what modules you want to train LoRA-layers. In the 
example above, it is using the attention layers (`.attn`). Currently, only linear layers can be lora'd. 
However, adding different layers (like convolutions) is possible as well. <br>
You can also set the `rank` and if you want to learn a new token for your training. The latter can be done by setting
`train_tokens` which expects a list of two things for each element: the token you want to add and a regex for 
the token / tokens that you want to use for initializing your new token.

## Image Reconstruction Training