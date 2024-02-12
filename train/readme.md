# Training
<p align="center">
    <img src="../figures/collage_3.jpg" width="600">
</p>

This directory provides a training code for Stable Cascade, as well as guides to download the models you need.
Specifically, you can find training scripts for the following use-cases:
- Text-to-Image
- ControlNet
- LoRA
- Image Reconstruction

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
<br><br>
There are a few more specifics to each kind of training and to datasets in general. These will be discussed below.

## Starting a Training
You can start an actual training very easily by first moving to the root directory of this repository (so [here](..)).
Next, the python command looks like the following:
```python
python3 training_file training_config
```
For example, if you want to train a LoRA model, the command would look like this:
```python
python3 train/train_c_lora.py configs/training/finetune_c_3b_lora.yaml
```

Moreover, we also provide a [bash script](example_train.sh) for working with slurm. Note, this assumes you have access to a cluster
that runs slurm as the cluster manager.

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

## Starting from a Pretrained Model
If you want to finetune any model you need the pretrained models. You can find details on how to download them in the
[models](../models) section. After downloading them, you need to modify the checkpoint paths in the config file too.
See below for example config files.

## Text-to-Image Training
You can use the following configs for finetuning Stage C on your own datasets. All necessary parameters were already
explained above. So there is nothing new here. Take a look at the config for finetuning the 
[3.6B Stage C](../configs/training/finetune_c_3b.yaml) and the [1B Stage C](../configs/training/finetune_c_1b.yaml).

## ControlNet Training
Training a ControlNet requires setting some extra parameters as well as adding the specific ControlNet Filter you want.
With filter, we simply mean a class that for example performs Canny Edge Detection, Human Pose Detection, etc.
```yaml
controlnet_blocks: [0, 4, 8, 12, 51, 55, 59, 63]
controlnet_filter: CannyFilter
controlnet_filter_params: 
  resize: 224
```
Here we need to give a little more detail on how Stage C's architecture looks like. It basically is just a stack of
residual blocks (convolutional and attention) that all work at the same latent resolution. We **do not** use a UNet. 
And this is where `controlnet_blocks` comes in. It determines at which blocks you want to inject the controlling 
information. This way, the ControlNet architecture differs from the common one used in Stable Diffusion where you
create an entire copy of the encoder of the UNet. With Stable Cascade it is a bit simpler and comes with the great 
benefit of using much fewer parameters. <br>
Next you define the class that filters the images and extracts the information you want to condition Stage C on
(Canny Edge Detection, Human Pose Detection, etc.) with the `controlnet_filter` parameter. In the example, we use the 
CannyFilter defined in the [controlnet.py](../modules/controlnet.py) file. This is the place where you can add your own 
ControlNet Filters. Lastly, `controlnet_filter_params` simply sets additional parameters to your `controlnet_filter`
class. That's it. You can view the example ControlNet configs for 
[Inpainting / Outpainting](../configs/training/controlnet_c_3b_inpainting.yaml), 
[Face Identity](../configs/training/controlnet_c_3b_identity.yaml), 
[Canny](../configs/training/controlnet_c_3b_canny.yaml) and 
[Super Resolution](../configs/training/controlnet_c_3b_sr.yaml).

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
You can also set the `rank` and if you want to learn a specific token for your training. The latter can be done by 
setting `train_tokens` which expects a list of two things for each element: the token you want to train and a regex for 
the token / tokens that you want to use for initializing the token. In the example above, a token `[fernando]` is
created and is initialized with the average of all tokens that include the word `dog`. Note, in order to **add** a new
token, **it has to start with `[` and end with `]`**. There is also the option of using existing tokens which will be
trained. For this, you just enter the token, **without** placing `[ ]` around it, like in the commented example above
for the token `sanil`. The second element is `null`, because we don't initialize this token and just finetune the
`snail` token. <br>
You can find an example config for training a LoRA [here](../configs/training/finetune_c_3b_lora.yaml).
Additionally, you can also download an 
[example dataset](https://huggingface.co/dome272/stable-cascade/blob/main/fernando.tar) for a cute little good boy dog. 
Simply download it and set the path in the config file to your destination path.

## Image Reconstruction Training
Here we mainly focus on training **Stage B**, because it is doing most of the heavy lifting for the compression, while
Stage A only applies a very small compression and thus the results are near perfect. Why do we use Stage A even? The
reason is just to make the training and inference of Stage B cheaper and faster. With Stage A in place, Stage B works
at a 4x smaller space (for example `1 x 4 x 256 x 256` instead of `1 x 3 x 1024 x 1024`). Furthermore, we observed that
Stage B learns faster when using Stage A compared to learning Stage B directly at pixel space. Anyway, why would you
even want to train Stage B? Either you want to try to create an even higher compression or finetune on something 
very specific. But this probably is a rare occasion. If you do want to, you can take a look at the training config 
for the large Stage B [here](../configs/training/finetune_b_3b.yaml) or for the small Stage B 
[here](../configs/training/finetune_b_700m.yaml). 

## Remarks
The codebase is in early development. You might encounter unexpected errors or not perfectly optimized training and
inference code. We apologize for that in advance. If there is interest, we will continue releasing updates to it,
aiming to bring in the latest improvements and optimizations. Moreover, we would be more than happy to receive
ideas, feedback or even updates from people that would like to contribute. Cheers.