# One image for every machine. Pick the flavour at build time:
#   GPU=cpu     robot laptop (ARM or x86): record, teleop, run a policy
#   GPU=nvidia  training on an NVIDIA card
#   GPU=rocm    training on an AMD Radeon card
FROM python:3.12-slim

ARG GPU=cpu
ARG LEROBOT_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg build-essential libgl1 libglib2.0-0 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LEROBOT_REF}" https://github.com/huggingface/lerobot.git /lerobot
WORKDIR /lerobot
RUN pip install --no-cache-dir -e ".[feetech,dataset,training]"

# Swap PyTorch for the right build (default PyPI build is NVIDIA's)
RUN set -e; \
    TV=$(python -c "import torch; print(torch.__version__.split('+')[0])"); \
    VV=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])"); \
    case "$GPU" in \
      nvidia) echo "Keeping NVIDIA PyTorch $TV" ;; \
      cpu) pip install --no-cache-dir --force-reinstall --no-deps torch==$TV torchvision==$VV \
             --index-url https://download.pytorch.org/whl/cpu \
           || { echo "No CPU-only build found, keeping default PyTorch"; GPU=nvidia; } ;; \
      rocm) ok=0; for r in rocm7.2 rocm7.1 rocm7.0 rocm6.4; do \
              if pip install --no-cache-dir --force-reinstall torch==$TV torchvision==$VV \
                   --index-url https://download.pytorch.org/whl/$r; then ok=1; break; fi; \
            done; [ $ok = 1 ] || { echo "No ROCm build for torch $TV"; exit 1; } ;; \
      *) echo "GPU must be cpu, nvidia or rocm"; exit 1 ;; \
    esac; \
    if [ "$GPU" != nvidia ]; then \
      pip uninstall -y $(pip list --format=freeze | grep -iE '^(nvidia-|cuda-)' | cut -d= -f1) 2>/dev/null || true; \
    fi; \
    pip install --no-cache-dir "numpy>=2.0,<2.3"; \
    python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"

# Calibration files and the Hugging Face login live here (mounted from the host)
ENV HF_HOME=/root/.cache/huggingface
WORKDIR /work
