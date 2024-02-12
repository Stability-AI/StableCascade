# Generic Diffusion Framework (GDF)

# Basic usage
GDF is a simple framework for working with diffusion models. It implements most common diffusion frameworks (DDPM / DDIM
, EDM, Rectified Flows, etc.) and makes it very easy to switch between them or combine different parts of different
frameworks

Using GDF is very straighforward, first of all just define an instance of the GDF class:

```python
from gdf import GDF
from gdf import CosineSchedule
from gdf import VPScaler, EpsilonTarget, CosineTNoiseCond, P2LossWeight

gdf = GDF(
    schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
    input_scaler=VPScaler(), target=EpsilonTarget(),
    noise_cond=CosineTNoiseCond(),
    loss_weight=P2LossWeight(),
)
```

You need to define the following components: 
* **Train Schedule**: This will return the logSNR schedule that will be used during training, some of the schedulers can be configured. A train schedule will then be called with a batch size and will randomly sample some values from the defined distribution.
* **Sample Schedule**: This is the schedule that will be used later on when sampling. It might be different from the training schedule. 
* **Input Scaler**: If you want to use Variance Preserving or LERP (rectified flows)
* **Target**: What the target is during training, usually: epsilon, x0 or v
* **Noise Conditioning**: You could directly pass the logSNR to your model but usually a normalized value is used instead, for example the EDM framework proposes to use `-logSNR/8`
* **Loss Weight**: There are many proposed loss weighting strategies, here you define which one you'll use

All of those classes are actually very simple logSNR centric definitions, for example the VPScaler is defined as just:
```python 
class VPScaler():
    def __call__(self, logSNR): 
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        return a, b

```

So it's very easy to extend this framework with custom schedulers, scalers, targets, loss weights, etc...

### Training

When you define your training loop you can get all you need by just doing:
```python
shift, loss_shift = 1, 1 # this can be set to higher values as per what the Simple Diffusion paper sugested for high resolution
for inputs, extra_conditions in dataloader_iterator:
	noised, noise, target, logSNR, noise_cond, loss_weight = gdf.diffuse(inputs, shift=shift, loss_shift=loss_shift) 
	pred = diffusion_model(noised, noise_cond, extra_conditions)

	loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
	loss_adjusted = (loss * loss_weight).mean()

	loss_adjusted.backward()
	optimizer.step()
	optimizer.zero_grad(set_to_none=True)
```

And that's all, you have a diffusion model training, where it's very easy to customize the different elements of the 
training from the GDF class.

### Sampling

The other important part is sampling, when you want to use this framework to sample you can just do the following:

```python
from gdf import DDPMSampler

shift = 1
sampling_configs = {
	"timesteps": 30, "cfg": 7,  "sampler": DDPMSampler(gdf), "shift": shift,
	"schedule": CosineSchedule(clamp_range=[0.0001, 0.9999])
}

*_, (sampled, _, _) = gdf.sample(
	diffusion_model, {"cond": extra_conditions}, latents.shape, 
	unconditional_inputs= {"cond": torch.zeros_like(extra_conditions)}, 
	device=device, **sampling_configs
)
```

# Available modules

TODO
