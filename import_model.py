from diffusers import StableDiffusionXLPipeline, ControlNetModel, AutoencoderKL

import bentoml


MODELS = {
    "controlnet": (
        ControlNetModel,
        "diffusers/controlnet-canny-sdxl-1.0",
        "sdxl-controlnet",
    ),
    "vae": (
        AutoencoderKL,
        "madebyollin/sdxl-vae-fp16-fix",
        "sdxl-controlnet-vae",
    ),
    "base": (
        StableDiffusionXLPipeline,
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-controlnet-base",
    ),
}


def import_models(force_redownload=False):

    for model_class, model_id, saved_model_name in MODELS.values():
        if not force_redownload:
            try:
                bentoml.models.get(saved_model_name)
                continue
            except bentoml.exceptions.NotFound:
                pass

        with bentoml.models.create(saved_model_name) as bento_model:
            model_class.from_pretrained(model_id).save_pretrained(bento_model.path)


if __name__ == "__main__":
    import_models()
