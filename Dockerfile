# Use an official Python runtime as a base image
FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app

RUN conda create -n mlops python=3.8
RUN echo "conda activate mlops" >> ~/.bashrc
ENV PATH /opt/conda/envs/mlops/bin:$PATH
RUN /bin/bash -c "source ~/.bashrc"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]