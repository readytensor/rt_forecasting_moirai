# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as builder

# Avoid prompts from apt and set python environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz \
    && tar -xzf Python-3.11.0.tgz \
    && cd Python-3.11.0 \
    && ./configure --enable-optimizations \
    && make -j 8 \
    && make altinstall

# Cleanup the source
RUN rm -rf Python-3.11.0.tgz Python-3.11.0

# Install and upgrade pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && python3.11 -m pip install --upgrade pip \
    && rm get-pip.py


# Add a symbolic link to python3 (optional)
RUN ln -s /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ln -s /usr/local/bin/python3.11 /usr/local/bin/python

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /opt/
RUN python3.11 -m pip install --no-cache-dir -r /opt/requirements.txt


# Copy model config file and model downloading script into the image
COPY ./src/config/model_config.json /opt/src/config/model_config.json
COPY ./src/prediction/download_model.py /opt/src/prediction/download_model.py
# Download the intended model - we are caching the model in the image
RUN python /opt/src/prediction/download_model.py

# Copy entry point and fix_line_endings scripts into the image
COPY ./entry_point.sh ./fix_line_endings.sh /opt/
# Set permissions and run fix_line_endings script
RUN chmod +x /opt/entry_point.sh /opt/fix_line_endings.sh \
    && /opt/fix_line_endings.sh "/opt/src" \
    && /opt/fix_line_endings.sh "/opt/entry_point.sh" \
    && chown -R 1000:1000 /opt \
    && chmod -R 777 /opt

# Copy source code into image and set permissions
COPY src /opt/src

# Set working directory
WORKDIR /opt/src

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/src:${PATH}" \
    TORCH_HOME="/opt" \
    MPLCONFIGDIR="/opt" \
    HF_HOME="/opt"

# Set non-root user and set entrypoint
USER 1000
ENTRYPOINT ["/opt/entry_point.sh"]
