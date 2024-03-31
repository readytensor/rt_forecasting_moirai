# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as builder

# Avoid prompts from apt and set python environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/src:${PATH}" \
    TORCH_HOME="/opt" \
    MPLCONFIGDIR="/opt" \
    TRANSFORMERS_CACHE="/opt" \
    HF_HOME="/opt"

# Install OS dependencies, Git, Python, and Pip in one layer to reduce image size and build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    dos2unix \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.9 python3-pip \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && python3.9 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy source code into image and set permissions
COPY src /opt/src
COPY ./entry_point.sh ./fix_line_endings.sh /opt/

# Copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt


# Copy model config file and model downloading script into the image
COPY ./src/config/model_config.json /opt/src/config/model_config.json
COPY ./src/prediction/download_model.py /opt/src/prediction/download_model.py

# Download the intended model - we are caching the model in the image
RUN python /opt/src/prediction/download_model.py

WORKDIR /opt/
COPY ./pyproject.toml  /opt/
RUN python3.9 -m pip install -e '.[notebook]'





RUN chmod +x /opt/entry_point.sh /opt/fix_line_endings.sh \
    && /opt/fix_line_endings.sh "/opt/src" \
    && /opt/fix_line_endings.sh "/opt/entry_point.sh" \
    && chown -R 1000:1000 /opt \
    && chmod -R 777 /opt

# Set working directory
WORKDIR /opt/src

# Set non-root user and set entrypoint
USER 1000
ENTRYPOINT ["/opt/entry_point.sh"]
