FROM python:3.9-slim
WORKDIR /home/biolib/

# Put noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip unzip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Torch requirements
COPY requirements_esm.txt .
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_esm.txt && \
    rm -rf /root/.cache/pip

# General requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    rm -rf /root/.cache/pip

# data record
COPY data_record.zip data_record.zip
RUN apt-get install -y unzip
RUN unzip data_record.zip
RUN rm data_record.zip

RUN mkdir -p output/
RUN mkdir -p data/
COPY data/cmap2.pkl data/cmap2.pkl
COPY data/input.fasta data/input.fasta
COPY src/ src/
COPY model/ model/
COPY run.py run.py
COPY run.sh run.sh