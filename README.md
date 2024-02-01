This project demonstrates how to build a ControlNet application using BentoML, powered by [diffusers](https://github.com/huggingface/diffusers).

## Prerequisites

- You have installed Python 3.9+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone -b 1.2 https://github.com/bentoml/BentoControlNet.git
cd BentoControlNet
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T09:43:40+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:APIService" can be accessed at http://localhost:3000/metrics.
2024-01-18T09:43:41+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:APIService" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
    'http://localhost:3000/generate' \
    -H 'accept: image/*' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@example-image.png;type=image/png' \
    -F 'params={
    "prompt": "A young man walking in a park, wearing jeans.",
    "negative_prompt": "ugly, disfigured, ill-structured, low resolution",
    "controlnet_conditioning_scale": 0.5,
    "num_inference_steps": 25
    }'
```

BentoML client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.generate(
        image=Path("example-image.png"),
        params={
                "prompt": "A young man walking in a park, wearing jeans.",
                "negative_prompt": "ugly, disfigured, ill-structure, low resolution",
                "controlnet_conditioning_scale": 0.5,
                "num_inference_steps": 25
        },
    )
```

## Deploy to production

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. A configuration YAML file (`bentofile.yaml`) is used to define the build options for your application. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/1.2/bentocloud/how-tos/manage-access-token.html), then run the following command in your project directory to deploy the application to BentoCloud.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: Alternatively, you can use BentoML to generate a [Docker image](https://docs.bentoml.com/en/1.2/guides/containerization.html) for a custom deployment.
