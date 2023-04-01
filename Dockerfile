FROM ubuntu:bionic
#FROM lurad101/default_denoise:latest


RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install lightning[extra]
