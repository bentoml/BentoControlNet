## SDXL with ControlNet example

### Prerequisites

You have installed Python 3.8 (or later) and `pip`.

### Set up the environment

For dependency isolation, we suggest you create a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies.

```bash
pip install -U pip && pip install -r requirements.txt
```

### Start a local ControlNet server

Run the script to download the models.

```bash
python import_model.py
```

The models are saved to the BentoML Model Store, a centralized repository for managing models. Run `bentoml models list` to view all available models locally.

Run `bentoml serve` to start a server locally. The server is accessible at http://0.0.0.0:3000.

### Build a Bento

A [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) in BentoML is a deployable artifact including model reference, data files, and dependencies. Once a Bento is built, you can containerize it as a Docker image or distribute it on [BentoCloud](https://www.bentoml.com/cloud) for better management, scalability and observability.

The `bentofile.yaml` file required to build a Bento is already available in the project directory with some basic configurations, while you can also customize it as needed. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

To build a Bento, run:

```bash
$ bentoml build
```

To build a Docker image for the Bento, run:

```bash
bentoml containerize BENTO_TAG
```

To push the Bento to BentoCloud, [log in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html) first and run the following command:

```bash
bentoml push BENTO_TAG
```

You can then [deploy the Bento](https://docs.bentoml.com/en/latest/bentocloud/how-tos/deploy-bentos.html) on BentoCloud.
