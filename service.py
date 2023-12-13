from __future__ import annotations

import typing as t

import cv2
import numpy as np
import PIL
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON, Image, Multipart


if t.TYPE_CHECKING:
    from PIL.Image import Image as PIL_Image
    from numpy.typing import NDArray

controlnet_model = bentoml.models.get("sdxl-controlnet")
base_model = bentoml.models.get("sdxl-controlnet-base")
vae_model = bentoml.models.get("sdxl-controlnet-vae")

class SDXLControlNetRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'cpu')
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model.path,
            torch_dtype=self.dtype,
        )

        self.vae = AutoencoderKL.from_pretrained(
            vae_model.path,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model.path,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=self.dtype
        ).to(self.device)


    @bentoml.Runnable.method(batchable=False)
    def generate(
            self,
            prompt: str,
            arr: NDArray,
            **kwargs,
    ):
        image = PIL.Image.fromarray(arr)
        return self.pipe(prompt, image=image, **kwargs).to_tuple()


runner = bentoml.Runner(
    SDXLControlNetRunnable,
    name=f"sdxl-controlnet-runner",
    models=[controlnet_model, base_model, vae_model]
)

svc = bentoml.Service(f"sdxl-controlnet-service", runners=[runner])

class Params(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str]
    controlnet_conditioning_scale: float = 0.5

params_sample = Params(
    prompt="aerial view, a futuristic bento box in a bright foggy jungle, hard lighting",
    negative_prompt="low quality, bad quality, sketches",
)

@svc.api(
    route="/generate",
    input=Multipart(image=Image(), params=JSON.from_sample(params_sample)),
    output=Image(),
)
async def generate(image: PIL_Image, params: Params):
    arr = np.array(image)
    arr = cv2.Canny(arr, 100, 200)
    arr = arr[:, :, None]
    arr = np.concatenate([arr, arr, arr], axis=2)
    params_d = params.dict()
    prompt = params_d.pop("prompt")
    res = await runner.generate.async_run(
        prompt,
        arr=arr,
        **params_d
    )
    return res[0][0]
