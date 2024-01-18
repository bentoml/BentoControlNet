This document demonstrates how to build a ControlNet application using BentoML, powered by [diffusers](https://github.com/huggingface/diffusers).

## **Prerequisites**

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See Installation for details.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T09:43:40+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:APIService" can be accessed at http://localhost:3000/metrics.
2024-01-18T09:43:41+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:APIService" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://0.0.0.0:3000](http://0.0.0.0:3000/). You can interact with it using Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://0.0.0.0:3000/generate' \
  -H 'accept: image/*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@hf-logo.png;type=image/png' \
  -F 'params={
  "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
  "negative_prompt": "low quality, bad quality, sketches",
  "controlnet_conditioning_scale": 0.5,
  "num_inference_steps": 25
}'

```

## Deploy the application to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. A configuration YAML file (`bentofile.yaml`) is used to define the build options for your application. It is used for packaging your application into a Bento. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

Make sure you have logged in to BentoCloud, then run the following command in your project directory to deploy the application to BentoCloud. Under the hood, this commands automatically builds a Bento, push the Bento to BentoCloud, and deploy it on BentoCloud.

```bash
bentoml deploy .
```

**Note**: Alternatively, you can manually build the Bento, containerize the Bento as a Docker image, and deploy it in any Docker-compatible environment. See [Docker deployment](https://docs.bentoml.org/en/latest/concepts/deploy.html#docker) for details.

Once the application is up and running on BentoCloud, you can access it via the exposed URL.
