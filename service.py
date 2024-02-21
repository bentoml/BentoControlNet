from __future__ import annotations

import typing as t

import numpy as np
import PIL
from PIL.Image import Image as PIL_Image

from pydantic import BaseModel

import bentoml

CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


@bentoml.service(
    traffic={"timeout": 600},
    workers=1,
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    }
)
class SDXLControlNetService:

    def __init__(self) -> None:

        import torch
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=self.dtype,
        )

        self.vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_ID,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=self.dtype
        ).to(self.device)


    @bentoml.api
    async def generate(
            self,
            prompt: str,
            arr: np.ndarray[t.Any, np.uint8],
            **kwargs,
    ):
        image = PIL.Image.fromarray(arr)
        return self.pipe(prompt, image=image, **kwargs).to_tuple()


class Params(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str]
    controlnet_conditioning_scale: float = 0.5
    num_inference_steps: int = 25


@bentoml.service(
    traffic={"timeout": 600},
    workers=8,
resources={"cpu": "1"}
)
class ControlNet:
    controlnet_service = bentoml.depends(SDXLControlNetService)

    @bentoml.api
    async def generate(self, image: PIL_Image, params: Params) -> PIL_Image:
        import cv2

        arr = np.array(image)
        arr = cv2.Canny(arr, 100, 200)
        arr = arr[:, :, None]
        arr = np.concatenate([arr, arr, arr], axis=2)
        params_d = params.dict()
        prompt = params_d.pop("prompt")
        res = await self.controlnet_service.generate(
            prompt,
            arr=arr,
            **params_d
        )
        return res[0][0]
