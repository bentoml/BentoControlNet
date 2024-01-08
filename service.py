from __future__ import annotations

import typing as t

import cv2
import numpy as np
import PIL
from PIL.Image import Image as PIL_Image

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from pydantic import BaseModel

import bentoml


@bentoml.service(traffic={"timeout": 600}, workers=1, resources={"gpu": "1"})
class SDXLControlNetService(bentoml.Runnable):
    controlnet_model_ref = bentoml.models.get("sdxl-controlnet")
    base_model_ref = bentoml.models.get("sdxl-controlnet-base")
    vae_model_ref = bentoml.models.get("sdxl-controlnet-vae")

    def __init__(self) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_ref.path,
            torch_dtype=self.dtype,
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.vae_model_ref.path,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model_ref.path,
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

params_sample = Params(
    prompt="aerial view, a futuristic bento box in a bright foggy jungle, hard lighting",
    negative_prompt="low quality, bad quality, sketches",
)


@bentoml.service(
    name="sdxl-controlnet-service",
    traffic={"timeout": 600},
    workers=8,
resources={"cpu": "1"}
)
class APIService:
    controlnet_service: SDXLControlNetService = bentoml.depends(SDXLControlNetService)

    @bentoml.api
    async def generate(self, image: PIL_Image, params: Params) -> PIL_Image:
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
