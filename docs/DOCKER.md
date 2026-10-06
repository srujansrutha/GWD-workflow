# Running the Wheat Advisor with Docker

This guide explains how to run the app and its language model in Docker, what every command does, and what to do when something goes wrong. No Python, CUDA or Ollama install is needed on your computer, only Docker.

## 1. What runs

```text
 your browser ──► http://localhost:8501
                        │
              ┌─────────▼─────────┐        ┌───────────────────┐
              │  app              │ ─────► │  ollama           │
              │  Streamlit, YOLO, │  asks  │  local LLM server │
              │  LangGraph        │        │  (writes report)  │
              └─────────┬─────────┘        └─────────▲─────────┘
                        │ reads                      │ downloads the model once
                  ./weights/*.pt              ┌──────┴────────┐
                  (your folder)               │  ollama-pull  │  runs once, then exits
                                              └───────────────┘
```

| Container | Kind | What it does |
| --- | --- | --- |
| `app` | service, keeps running | The two pages (Head counter, Field report), the YOLO11 detector and the LangGraph workflow |
| `ollama` | service, keeps running | The local language model server. Models are stored in a Docker volume |
| `ollama-pull` | one-time job | Downloads the report model (`qwen3.5:9b`, about 6.6 GB) if it is missing, then exits |

Three kinds of data live **outside** the app image, so the image stays small and nothing private is baked into it:

| Data | Where it lives | Why |
| --- | --- | --- |
| Detector weights (`.pt`) | the `weights/` folder on your computer, mounted read-only | Model files are not in Git, and you can swap them without rebuilding |
| Report model (LLM) | Docker volume `wheat-advisor_ollama-models` | A 6.6 GB download should survive deleting containers |
| Weather, forecast and place-name cache | Docker volume `wheat-advisor_advisor-cache` | Saves repeat calls to the online services |

## 2. Before you start

You need:

- **Docker**: Docker Desktop (Windows, macOS) or Docker Engine with the Compose plugin (Linux). Check with `docker compose version`.
- **About 20 GB of free disk**: the app image is about 3 GB, the Ollama image 9.4 GB and the model 6.6 GB. The GPU image is larger because it carries the CUDA libraries.
- **The detector weights** at `weights/wheat_yolo11s_gwhd21.pt`. They are not in the repository. Train them with the notebooks ([TRAINING.md](TRAINING.md#10-reproducing-the-project)) or use your own YOLO weights. Without them the app starts but shows "No model weights found".
- **For GPU mode only**: an NVIDIA GPU with a recent driver and GPU support in Docker (Docker Desktop with WSL 2 on Windows, or the NVIDIA Container Toolkit on Linux).

## 3. There is no ready-made image to download

The image is **not** published to Docker Hub or any other registry, so `docker pull` has nothing to fetch. Docker **builds** the image on your computer from this repository, using [docker/app.Dockerfile](../docker/app.Dockerfile). The first build takes a while because PyTorch is a large download; later builds reuse what was already downloaded. The Ollama containers do use a ready-made public image (`ollama/ollama`), which Docker pulls for you.

## 4. Quick start

```bash
git clone https://github.com/srujansrutha/GWD-workflow.git
cd GWD-workflow

# put your detector weights in place
mkdir -p weights && cp /path/to/wheat_yolo11s_gwhd21.pt weights/

# build and start everything (pick ONE of these two, see section 5)
docker compose up --build                                          # CPU: works on any computer
docker compose -f compose.yaml -f compose.gpu.yaml up --build      # NVIDIA GPU
```

Then open <http://localhost:8501>.

On the first start Docker builds the image and downloads the Ollama image and the report model, so it can take a long time on a slow connection. The app is already usable while the model downloads: the Field report shows "model not ready" and writes the rule-based wording until the model is there. Later starts take seconds.

Leave the terminal open to watch the logs; press `Ctrl+C` to stop. To run in the background instead, add `-d` (`docker compose up -d --build`), and the containers then start again whenever Docker starts, until you run `docker compose down`.

## 5. Which command do I run?

You choose. Docker does **not** try the GPU setup first and fall back by itself.

| Situation | Command |
| --- | --- |
| Any computer, no NVIDIA GPU, or you just want to try it | `docker compose up --build` |
| Computer with an NVIDIA GPU and GPU support in Docker | `docker compose -f compose.yaml -f compose.gpu.yaml up --build` |
| You already run Ollama on your computer and want to use it | see below |

[compose.gpu.yaml](../compose.gpu.yaml) is not a complete setup on its own. It only holds the differences (the CUDA version of PyTorch and the GPU request for the two containers), and Docker merges it on top of [compose.yaml](../compose.yaml). It is a separate file because a GPU request makes Docker refuse to start on a computer with no NVIDIA GPU, so the plain command has to stay free of it. The CPU image is tagged `wheat-advisor-app:cpu` and the GPU image `wheat-advisor-app:gpu`, so building one never overwrites the other.

Without a GPU the detector still works, but the report model is much slower, because a 9-billion-parameter model is meant to run on a GPU.

**Using Ollama that is already running on your computer** (skips the two Ollama containers and their 6.6 GB download):

```bash
ADVISOR_OLLAMA_URL=http://host.docker.internal:11434 docker compose up --build --no-deps app
```

```powershell
$env:ADVISOR_OLLAMA_URL="http://host.docker.internal:11434"; docker compose up --build --no-deps app
```

`--no-deps` means "start only the app, not the containers it normally waits for". If the app says Ollama is not running, your Ollama is probably listening on `localhost` only. Start it with `OLLAMA_HOST=0.0.0.0` (or turn on "Expose Ollama to the network" in its settings). Ollama has no password, so do this only behind a firewall.

## 6. What each command does

| Command | What it does |
| --- | --- |
| `docker compose up --build` | Builds the app image if needed, then starts the containers and shows their logs |
| `docker compose up -d --build` | The same, but in the background |
| `docker compose ps -a` | Shows each container and whether it is healthy. `app` and `ollama` should say `healthy`; `ollama-pull` finishes with `exited (0)`, which is success (`-a` is needed to list finished containers) |
| `docker compose logs -f app` | Follows the app's log. Use `ollama` or `ollama-pull` to see those instead. `Ctrl+C` stops following, not the app |
| `docker compose exec ollama ollama list` | Lists the models stored in the Ollama container |
| `docker compose exec ollama ollama pull qwen3.5:9b` | Updates the report model on purpose. It is never updated by itself |
| `docker compose down` | Stops and removes the containers. The model and cache volumes stay, so the next start is fast |
| `docker compose down -v` | Also deletes the volumes: the downloaded model and the cache. Use it to free about 7 GB |
| `docker compose build --pull` | Rebuilds the app image with the newest base image, which picks up security fixes |
| `git pull` then `docker compose up --build` | Updates to the newest code. Only the code layer rebuilds, so it is quick |

To free the rest of the disk after `down -v`: `docker image rm wheat-advisor-app:cpu` (or `:gpu`) removes the app image, and `docker image prune` removes unused leftovers.

## 7. Settings you can change

Set these in the shell before the command (bash: `NAME=value docker compose up`, PowerShell: `$env:NAME="value"; docker compose up`). None are needed for a normal start.

| Setting | Default | What it does |
| --- | --- | --- |
| `APP_PORT` | `8501` | The port on your computer. Use it when 8501 is taken: `APP_PORT=8502 docker compose up` |
| `ADVISOR_MODEL` | `qwen3.5:9b` | The Ollama model used for the report. A model that is not downloaded yet is downloaded on the next start |
| `ADVISOR_OLLAMA_URL` | `http://ollama:11434` | Where the app finds Ollama. Change it only to use an Ollama outside Docker |
| `TORCH_CUDA_INDEX` | CUDA 13 wheels | GPU build only. For an older NVIDIA driver use `https://download.pytorch.org/whl/cu128` |

## 8. When something goes wrong

| What you see | What it means | What to do |
| --- | --- | --- |
| "Cannot connect to the Docker daemon" | Docker itself is not running | Start Docker Desktop and wait until it says it is running |
| An error about the `nvidia` device driver or GPU when you used the GPU command | This computer has no NVIDIA GPU, or Docker has no GPU support | Use the plain command: `docker compose up --build` |
| "No model weights found" in the app | There is no file at `weights/wheat_yolo11s_gwhd21.pt` | Put the weights there. No rebuild is needed, only a page reload |
| The report says "model not ready" or "Ollama is not running" | The model is still downloading, or the Ollama container is not up | Run `docker compose ps -a` and `docker compose logs ollama-pull`. The report works with rule-based wording meanwhile |
| The build fails with "THESE PACKAGES DO NOT MATCH THE HASHES" | A slow or unstable connection corrupted a large download | Run the same command again. Downloads that finished are kept, so it resumes quickly |
| "port is already allocated" | Something else uses port 8501, for example the app run without Docker | Stop that one, or use `APP_PORT=8502` |
| The report is very slow | The model is running on the CPU | Use a computer with an NVIDIA GPU and the GPU command |

## 9. Safety choices in this setup

- The app's port is published to `127.0.0.1` only, so other computers on your network cannot reach the app (it has no login).
- The app runs as a normal user inside the container, not as root, and the image contains only what the app needs.
- The Ollama image is pinned to a fixed version (0.35.1), and the report model is only downloaded when it is missing, so a newer upstream version never replaces the tested one without you asking.
- The `.env` file, data, training runs and weights are excluded from the image by [.dockerignore](../.dockerignore).
- The Debian base image can carry known vulnerabilities before a fix exists. A Trivy scan of the CPU image found 59 high or critical findings in the Debian operating-system layer, and none of them had a fix available at that time. `docker compose build --pull` picks up fixes as Debian releases them.

## 10. What is not in Docker

Training and data preparation still run from the notebooks on your own machine ([TRAINING.md](TRAINING.md)). They are jobs that run once and finish, so they would need their own image and a command instead of a long-running service.

## 11. Files

| File | What it is |
| --- | --- |
| [compose.yaml](../compose.yaml) | The three containers, their ports, volumes and start order |
| [compose.gpu.yaml](../compose.gpu.yaml) | The GPU add-on, merged on top of `compose.yaml` |
| [docker/app.Dockerfile](../docker/app.Dockerfile) | The recipe for the app image |
| [requirements-app.txt](../requirements-app.txt) | Only the libraries the app needs (no Jupyter, Kaggle or training tools) |
| [.dockerignore](../.dockerignore) | What is kept out of the image |
