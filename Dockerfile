FROM python:3.9-slim
WORKDIR /home/biolib/

# Install needed dependencies
RUN apt-get update && \
    apt-get install -y python3-pip

# RUN pip3 install numpy==1.26

COPY requirements.txt .
RUN pip3 install -r requirements.txt

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