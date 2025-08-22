# ==============================================================================
# ADNet Development Environment - Multi-stage Dockerfile
# Base: Ubuntu 24.04 with CUDA 12.9 support
# Architecture: Base -> Development/Training/Production stages
# ==============================================================================

# ==============================================================================
# BASE STAGE - Common dependencies for all stages
# ==============================================================================
FROM ubuntu:24.04 AS base

# Build arguments for version management
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Image metadata for proper naming and versioning
LABEL org.opencontainers.image.title="adnet-base"
LABEL org.opencontainers.image.description="Autonomous Driving Neural Network Framework"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="ADNet Project"
LABEL org.opencontainers.image.authors="ADNet Development Team"
LABEL maintainer="adnet-dev@example.com"

# System setup - Core environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Define cleanup macro - Used after major installations to reduce layer size
ENV CLEANUP_CMD="apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip && find /usr/local -name '*.pyc' -delete"

# GPU environment variables - RTX 5070 Ti optimizations
ENV CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1

# Install system dependencies
RUN echo "Installing system packages..." && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    openssh-client \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    btop \
    tmux \
    screen \
    unzip \
    zip \
    tree \
    netcat-openbsd \
    iproute2 \
    net-tools \
    iputils-ping \
    sudo \
    ca-certificates \
    gnupg \
    libeigen3-dev \
    libopencv-dev \
    libboost-all-dev \
    libgtest-dev \
    libnvidia-compute-575 \
    libnvidia-decode-575 \
    libnvidia-encode-575 && \
    eval "$CLEANUP_CMD"

# Install CUDA toolkit
RUN echo "Installing CUDA toolkit..." && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-9 && \
    eval "$CLEANUP_CMD"

# Set CUDA environment
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Node.js
RUN echo "Installing Node.js..." && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    eval "$CLEANUP_CMD"

# Configure timezone
RUN echo "Configuring timezone..." && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Configure Python alternatives
RUN echo "Configuring Python alternatives..." && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create and activate virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Upgrade pip in virtual environment
RUN pip install --upgrade pip setuptools wheel

# Install starship prompt
RUN echo "Installing Starship prompt..." && \
    curl -sS https://starship.rs/install.sh | sh -s -- --yes && \
    echo '' >> ~/.bashrc && \
    echo '# Initialize Starship prompt' >> ~/.bashrc && \
    echo 'eval "$(starship init bash)"' >> ~/.bashrc && \
    eval "$CLEANUP_CMD"

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA support
RUN echo "Installing PyTorch..." && \
    pip install --no-cache-dir --break-system-packages \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129 && \
    eval "$CLEANUP_CMD"

# Install build dependencies
RUN echo "Installing build dependencies..." && \
    pip install --no-cache-dir --break-system-packages \
    packaging wheel setuptools cython numpy scipy && \
    eval "$CLEANUP_CMD"

# Install base dependencies
COPY requirements/base.txt requirements/base.txt
RUN echo "Installing base dependencies..." && \
    pip install --no-cache-dir --break-system-packages -r requirements/base.txt && \
    eval "$CLEANUP_CMD"

# Copy and install project
COPY . .
RUN echo "Installing project..." && \
    pip install --break-system-packages -e . && \
    eval "$CLEANUP_CMD"

# ==============================================================================
# DEVELOPMENT STAGE - Full development environment with Jupyter, testing tools
# ==============================================================================
FROM base AS development

# Stage-specific labels
LABEL org.opencontainers.image.title="adnet-dev"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL stage="dev"

# Install development dependencies
COPY requirements/dev.txt requirements/dev.txt
RUN echo "Installing development dependencies..." && \
    pip install --no-cache-dir --break-system-packages -r requirements/dev.txt && \
    eval "$CLEANUP_CMD"

# Install visualization and UI dependencies
COPY requirements/visualization.txt requirements/visualization.txt
RUN echo "Installing visualization dependencies..." && \
    pip install --no-cache-dir --break-system-packages -r requirements/visualization.txt && \
    eval "$CLEANUP_CMD"

# Install Claude Code CLI
RUN echo "Installing Claude Code..." && \
    npm install -g @anthropic-ai/claude-code && \
    eval "$CLEANUP_CMD"

# Configure Jupyter to work with the virtual environment
RUN echo "Configuring Jupyter..." && \
   jupyter notebook --generate-config && \
   echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
   echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
   echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
   echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports for development services
EXPOSE 8888 6006 8097 8000 7860 8080

CMD ["bash"]

# ==============================================================================
# TRAINING STAGE - Optimized for ML training workloads
# ==============================================================================
FROM base AS training

# Stage-specific labels
LABEL org.opencontainers.image.title="adnet-training"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL stage="training"

# Install training dependencies
COPY requirements/training.txt requirements/training.txt
RUN echo "Installing training dependencies..." && \
    pip install --no-cache-dir --break-system-packages -r requirements/training.txt && \
    eval "$CLEANUP_CMD"

# Training-specific optimizations
ENV TORCH_CUDNN_BENCHMARK=1

CMD ["python", "scripts/training/train.py"]

# ==============================================================================
# PRODUCTION STAGE - Minimal deployment image for API serving
# ==============================================================================
FROM base AS production

# Stage-specific labels
LABEL org.opencontainers.image.title="adnet-production"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL stage="production"

# Install API dependencies
COPY requirements/api.txt requirements/api.txt
RUN echo "Installing API dependencies..." && \
    pip install --no-cache-dir --break-system-packages -r requirements/api.txt && \
    eval "$CLEANUP_CMD"

# Create non-root user for production
RUN useradd -m -s /bin/bash adnet && \
   chown -R adnet:adnet /workspace /opt/venv

# Switch to non-root user
USER adnet

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
   CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["uvicorn", "adnet.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Build Instructions:
# docker build --target development -t adnet-dev:latest .
# docker build --target training -t adnet-train:latest .
# docker build --target production -t adnet-api:latest .
# ==============================================================================
