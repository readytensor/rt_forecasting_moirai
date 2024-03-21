# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as builder

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python and pip and add symbolic link to python3
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3-pip

RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt

# Copy model config file and model downloading script into the image
COPY ./src/config/model_config.json /opt/src/config/model_config.json
COPY ./src/prediction/download_model.py /opt/src/prediction/download_model.py
# Download the intended model - we are caching the model in the image
RUN python /opt/src/prediction/download_model.py

# copy src code into image and chmod scripts
COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"


# Set working directory
WORKDIR /opt/src


# set python variables and path
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
ENV TORCH_HOME="/opt"
ENV MPLCONFIGDIR="/opt"
ENV TRANSFORMERS_CACHE="/opt"


RUN chown -R 1000:1000 /opt
RUN chmod -R 777 /opt
# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
