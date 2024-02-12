import torch
import torchvision
import numpy as np
from torchtools.transforms import SmartCrop
import math

class Bucketeer():
    def __init__(self, dataloader, density=256*256, factor=8, ratios=[1/1, 1/2, 3/4, 3/5, 4/5, 6/9, 9/16], reverse_list=True, randomize_p=0.3, randomize_q=0.2, crop_mode='random', p_random_ratio=0.0, interpolate_nearest=False):
        assert crop_mode in ['center', 'random', 'smart']
        self.crop_mode = crop_mode
        self.ratios = ratios
        if reverse_list:
            for r in list(ratios):
                if 1/r not in self.ratios:
                    self.ratios.append(1/r)
        self.sizes = [(int(((density/r)**0.5//factor)*factor), int(((density*r)**0.5//factor)*factor)) for r in ratios]        
        self.batch_size = dataloader.batch_size
        self.iterator = iter(dataloader)
        self.buckets = {s: [] for s in self.sizes}
        self.smartcrop = SmartCrop(int(density**0.5), randomize_p, randomize_q) if self.crop_mode=='smart' else None
        self.p_random_ratio = p_random_ratio
        self.interpolate_nearest = interpolate_nearest

    def get_available_batch(self):
        for b in self.buckets:
            if len(self.buckets[b]) >= self.batch_size:
                batch = self.buckets[b][:self.batch_size]
                self.buckets[b] = self.buckets[b][self.batch_size:]
                return batch
        return None

    def get_closest_size(self, x):
        if self.p_random_ratio > 0 and np.random.rand() < self.p_random_ratio:
            best_size_idx = np.random.randint(len(self.ratios))
        else:
            w, h = x.size(-1), x.size(-2)
            best_size_idx = np.argmin([abs(w/h-r) for r in self.ratios])
        return self.sizes[best_size_idx]

    def get_resize_size(self, orig_size, tgt_size):
        if (tgt_size[1]/tgt_size[0] - 1) * (orig_size[1]/orig_size[0] - 1) >= 0:
            alt_min = int(math.ceil(max(tgt_size)*min(orig_size)/max(orig_size)))
            resize_size = max(alt_min, min(tgt_size))
        else:
            alt_max = int(math.ceil(min(tgt_size)*max(orig_size)/min(orig_size)))
            resize_size = max(alt_max, max(tgt_size))
        return resize_size

    def __next__(self):
        batch = self.get_available_batch()
        while batch is None:
            elements = next(self.iterator)
            for dct in elements:
                img = dct['images']
                size = self.get_closest_size(img)
                resize_size = self.get_resize_size(img.shape[-2:], size)
                if self.interpolate_nearest:
                    img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                else:
                    img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
                if self.crop_mode == 'center':
                    img = torchvision.transforms.functional.center_crop(img, size)
                elif self.crop_mode == 'random':
                    img = torchvision.transforms.RandomCrop(size)(img)
                elif self.crop_mode == 'smart':
                    self.smartcrop.output_size = size
                    img = self.smartcrop(img)
                self.buckets[size].append({**{'images': img}, **{k:dct[k] for k in dct if k != 'images'}})
            batch = self.get_available_batch()

        out = {k:[batch[i][k] for i in range(len(batch))] for k in batch[0]}
        return {k: torch.stack(o, dim=0) if isinstance(o[0], torch.Tensor) else o for k, o in out.items()}
