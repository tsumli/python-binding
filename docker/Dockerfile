FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

# install libraries
RUN apt update && apt upgrade -y
RUN apt install -y \
    git \
    wget \
    pkg-config \
    libssl-dev \
    curl \
    cmake \
    unzip \
    vim \
    libgl1-mesa-dev \
    libglib2.0-0  \
    ffmpeg \
    nodejs \
    npm \
    libcairo2-dev

RUN apt update

# install python
RUN apt install python3.10 python3-pip -y
RUN python3.10 -m pip install -U pip

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update
RUN rustup install nightly
RUN rustup component add clippy rls rust-analysis rust-src rust-docs rustfmt rust-analyzer

## install pip libraries
COPY docker/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN echo 'alias python="/usr/bin/python3"' >> ~/.bashrc
RUN echo 'alias pip="/usr/bin/python3 -m pip"' >> ~/.bashrc

