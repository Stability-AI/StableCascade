import torchvision
import torch
from torch import nn
import numpy as np
import kornia
import cv2
from core.utils import load_or_fail
from insightface.app.common import Face
from .effnet import EfficientNetEncoder
from .cnet_modules.pidinet import PidiNetDetector
from .cnet_modules.inpainting.saliency_model import MicroResNet
from .cnet_modules.face_id.arcface import FaceDetector, ArcFaceRecognizer
from .common import LayerNorm2d


class CNetResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.Sequential(
            LayerNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            LayerNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.blocks(x)


class ControlNet(nn.Module):
    def __init__(self, c_in=3, c_proj=2048, proj_blocks=None, bottleneck_mode=None):
        super().__init__()
        if bottleneck_mode is None:
            bottleneck_mode = 'effnet'
        self.proj_blocks = proj_blocks
        if bottleneck_mode == 'effnet':
            embd_channels = 1280
            self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
            if c_in != 3:
                in_weights = self.backbone[0][0].weight.data
                self.backbone[0][0] = nn.Conv2d(c_in, 24, kernel_size=3, stride=2, bias=False)
                if c_in > 3:
                    nn.init.constant_(self.backbone[0][0].weight, 0)
                    self.backbone[0][0].weight.data[:, :3] = in_weights[:, :3].clone()
                else:
                    self.backbone[0][0].weight.data = in_weights[:, :c_in].clone()
        elif bottleneck_mode == 'simple':
            embd_channels = c_in
            self.backbone = nn.Sequential(
                nn.Conv2d(embd_channels, embd_channels * 4, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embd_channels * 4, embd_channels, kernel_size=3, padding=1),
            )
        elif bottleneck_mode == 'large':
            self.backbone = nn.Sequential(
                nn.Conv2d(c_in, 4096 * 4, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(4096 * 4, 1024, kernel_size=1),
                *[CNetResBlock(1024) for _ in range(8)],
                nn.Conv2d(1024, 1280, kernel_size=1),
            )
            embd_channels = 1280
        else:
            raise ValueError(f'Unknown bottleneck mode: {bottleneck_mode}')
        self.projections = nn.ModuleList()
        for _ in range(len(proj_blocks)):
            self.projections.append(nn.Sequential(
                nn.Conv2d(embd_channels, embd_channels, kernel_size=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embd_channels, c_proj, kernel_size=1, bias=False),
            ))
            nn.init.constant_(self.projections[-1][-1].weight, 0)  # zero output projection

    def forward(self, x):
        x = self.backbone(x)
        proj_outputs = [None for _ in range(max(self.proj_blocks) + 1)]
        for i, idx in enumerate(self.proj_blocks):
            proj_outputs[idx] = self.projections[i](x)
        return proj_outputs


class ControlNetDeliverer():
    def __init__(self, controlnet_projections):
        self.controlnet_projections = controlnet_projections
        self.restart()

    def restart(self):
        self.idx = 0
        return self

    def __call__(self):
        if self.idx < len(self.controlnet_projections):
            output = self.controlnet_projections[self.idx]
        else:
            output = None
        self.idx += 1
        return output


# CONTROLNET FILTERS ----------------------------------------------------

class BaseFilter():
    def __init__(self, device):
        self.device = device

    def num_channels(self):
        return 3

    def __call__(self, x):
        return x


class CannyFilter(BaseFilter):
    def __init__(self, device, resize=224):
        super().__init__(device)
        self.resize = resize

    def num_channels(self):
        return 1

    def __call__(self, x):
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')
        edges = [cv2.Canny(x[i].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8), 100, 200) for i in range(len(x))]
        edges = torch.stack([torch.tensor(e).div(255).unsqueeze(0) for e in edges], dim=0)
        if self.resize is not None:
            edges = nn.functional.interpolate(edges, size=orig_size, mode='bilinear')
        return edges


class QRFilter(BaseFilter):
    def __init__(self, device, resize=224, blobify=True, dilation_kernels=[3, 5, 7], blur_kernels=[15]):
        super().__init__(device)
        self.resize = resize
        self.blobify = blobify
        self.dilation_kernels = dilation_kernels
        self.blur_kernels = blur_kernels

    def num_channels(self):
        return 1

    def __call__(self, x):
        x = x.to(self.device)
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')

        x = kornia.color.rgb_to_hsv(x)[:, -1:]
        # blobify
        if self.blobify:
            d_kernel = np.random.choice(self.dilation_kernels)
            d_blur = np.random.choice(self.blur_kernels)
            if d_blur > 0:
                x = torchvision.transforms.GaussianBlur(d_blur)(x)
            if d_kernel > 0:
                blob_mask = ((torch.linspace(-0.5, 0.5, d_kernel).pow(2)[None] + torch.linspace(-0.5, 0.5,
                                                                                                d_kernel).pow(2)[:,
                                                                                 None]) < 0.3).float().to(self.device)
                x = kornia.morphology.dilation(x, blob_mask)
                x = kornia.morphology.erosion(x, blob_mask)
        # mask
        vmax, vmin = x.amax(dim=[2, 3], keepdim=True)[0], x.amin(dim=[2, 3], keepdim=True)[0]
        th = (vmax - vmin) * 0.33
        high_brightness, low_brightness = (x > (vmax - th)).float(), (x < (vmin + th)).float()
        mask = (torch.ones_like(x) - low_brightness + high_brightness) * 0.5

        if self.resize is not None:
            mask = nn.functional.interpolate(mask, size=orig_size, mode='bilinear')
        return mask.cpu()


class PidiFilter(BaseFilter):
    def __init__(self, device, resize=224, dilation_kernels=[0, 3, 5, 7, 9], binarize=True):
        super().__init__(device)
        self.resize = resize
        self.model = PidiNetDetector(device)
        self.dilation_kernels = dilation_kernels
        self.binarize = binarize

    def num_channels(self):
        return 1

    def __call__(self, x):
        x = x.to(self.device)
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')

        x = self.model(x)
        d_kernel = np.random.choice(self.dilation_kernels)
        if d_kernel > 0:
            blob_mask = ((torch.linspace(-0.5, 0.5, d_kernel).pow(2)[None] + torch.linspace(-0.5, 0.5, d_kernel).pow(2)[
                                                                             :, None]) < 0.3).float().to(self.device)
            x = kornia.morphology.dilation(x, blob_mask)
        if self.binarize:
            th = np.random.uniform(0.05, 0.7)
            x = (x > th).float()

        if self.resize is not None:
            x = nn.functional.interpolate(x, size=orig_size, mode='bilinear')
        return x.cpu()


class SRFilter(BaseFilter):
    def __init__(self, device, scale_factor=1 / 4):
        super().__init__(device)
        self.scale_factor = scale_factor

    def num_channels(self):
        return 3

    def __call__(self, x):
        x = torch.nn.functional.interpolate(x.clone(), scale_factor=self.scale_factor, mode="nearest")
        return torch.nn.functional.interpolate(x, scale_factor=1 / self.scale_factor, mode="nearest")


class SREffnetFilter(BaseFilter):
    def __init__(self, device, scale_factor=1/2):
        super().__init__(device)
        self.scale_factor = scale_factor

        self.effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        self.effnet = EfficientNetEncoder().to(self.device)
        effnet_checkpoint = load_or_fail("models/effnet_encoder.safetensors")
        self.effnet.load_state_dict(effnet_checkpoint)
        self.effnet.eval().requires_grad_(False)

    def num_channels(self):
        return 16

    def __call__(self, x):
        x = torch.nn.functional.interpolate(x.clone(), scale_factor=self.scale_factor, mode="nearest")
        with torch.no_grad():
            effnet_embedding = self.effnet(self.effnet_preprocess(x.to(self.device))).cpu()
        effnet_embedding = torch.nn.functional.interpolate(effnet_embedding, scale_factor=1/self.scale_factor, mode="nearest")
        upscaled_image = torch.nn.functional.interpolate(x, scale_factor=1/self.scale_factor, mode="nearest")
        return effnet_embedding, upscaled_image


class InpaintFilter(BaseFilter):
    def __init__(self, device, thresold=[0.04, 0.4], p_outpaint=0.4):
        super().__init__(device)
        self.saliency_model = MicroResNet().eval().requires_grad_(False).to(device)
        self.saliency_model.load_state_dict(load_or_fail("modules/cnet_modules/inpainting/saliency_model.pt"))
        self.thresold = thresold
        self.p_outpaint = p_outpaint

    def num_channels(self):
        return 4

    def __call__(self, x, mask=None, threshold=None, outpaint=None):
        x = x.to(self.device)
        resized_x = torchvision.transforms.functional.resize(x, 240, antialias=True)
        if threshold is None:
            threshold = np.random.uniform(self.thresold[0], self.thresold[1])
        if mask is None:
            saliency_map = self.saliency_model(resized_x) > threshold
            if outpaint is None:
                if np.random.rand() < self.p_outpaint:
                    saliency_map = ~saliency_map
            else:
                if outpaint:
                    saliency_map = ~saliency_map
            interpolated_saliency_map = torch.nn.functional.interpolate(saliency_map.float(), size=x.shape[2:], mode="nearest")
            saliency_map = torchvision.transforms.functional.gaussian_blur(interpolated_saliency_map, 141) > 0.5
            inpainted_images = torch.where(saliency_map, torch.ones_like(x), x)
            mask = torch.nn.functional.interpolate(saliency_map.float(), size=inpainted_images.shape[2:], mode="nearest")
        else:
            mask = mask.to(self.device)
            inpainted_images = torch.where(mask, torch.ones_like(x), x)
        c_inpaint = torch.cat([inpainted_images, mask], dim=1)
        return c_inpaint.cpu()


# IDENTITY
class IdentityFilter(BaseFilter):
    def __init__(self, device, max_faces=4, p_drop=0.05, p_full=0.3):
        detector_path = 'modules/cnet_modules/face_id/models/buffalo_l/det_10g.onnx'
        recognizer_path = 'modules/cnet_modules/face_id/models/buffalo_l/w600k_r50.onnx'

        super().__init__(device)
        self.max_faces = max_faces
        self.p_drop = p_drop
        self.p_full = p_full

        self.detector = FaceDetector(detector_path, device=device)
        self.recognizer = ArcFaceRecognizer(recognizer_path, device=device)

        self.id_colors = torch.tensor([
            [1.0, 0.0, 0.0],  # RED
            [0.0, 1.0, 0.0],  # GREEN
            [0.0, 0.0, 1.0],  # BLUE
            [1.0, 0.0, 1.0],  # PURPLE
            [0.0, 1.0, 1.0],  # CYAN
            [1.0, 1.0, 0.0],  # YELLOW
            [0.5, 0.0, 0.0],  # DARK RED
            [0.0, 0.5, 0.0],  # DARK GREEN
            [0.0, 0.0, 0.5],  # DARK BLUE
            [0.5, 0.0, 0.5],  # DARK PURPLE
            [0.0, 0.5, 0.5],  # DARK CYAN
            [0.5, 0.5, 0.0],  # DARK YELLOW
        ])

    def num_channels(self):
        return 512

    def get_faces(self, image):
        npimg = image.permute(1, 2, 0).mul(255).to(device="cpu", dtype=torch.uint8).cpu().numpy()
        bgr = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
        bboxes, kpss = self.detector.detect(bgr, max_num=self.max_faces)
        N = len(bboxes)
        ids = torch.zeros((N, 512), dtype=torch.float32)
        for i in range(N):
            face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            ids[i, :] = self.recognizer.get(bgr, face)
        tbboxes = torch.tensor(bboxes[:, :4], dtype=torch.int)

        ids = ids / torch.linalg.norm(ids, dim=1, keepdim=True)
        return tbboxes, ids  # returns bounding boxes (N x 4) and ID vectors (N x 512)

    def __call__(self, x):
        visual_aid = x.clone().cpu()
        face_mtx = torch.zeros(x.size(0), 512, x.size(-2) // 32, x.size(-1) // 32)

        for i in range(x.size(0)):
            bounding_boxes, ids = self.get_faces(x[i])
            for j in range(bounding_boxes.size(0)):
                if np.random.rand() > self.p_drop:
                    sx, sy, ex, ey = (bounding_boxes[j] / 32).clamp(min=0).round().int().tolist()
                    ex, ey = max(ex, sx + 1), max(ey, sy + 1)
                    if bounding_boxes.size(0) == 1 and np.random.rand() < self.p_full:
                        sx, sy, ex, ey = 0, 0, x.size(-1) // 32, x.size(-2) // 32
                    face_mtx[i, :, sy:ey, sx:ex] = ids[j:j + 1, :, None, None]
                    visual_aid[i, :, int(sy * 32):int(ey * 32), int(sx * 32):int(ex * 32)] += self.id_colors[j % 13, :,
                                                                                              None, None]
                    visual_aid[i, :, int(sy * 32):int(ey * 32), int(sx * 32):int(ex * 32)] *= 0.5

        return face_mtx.to(x.device), visual_aid.to(x.device)
