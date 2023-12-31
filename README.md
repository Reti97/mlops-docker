# Train distilbert-base-uncased from Docker

This repository contains all the files needed to train a [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) Model. To install the needed dependencies, you can use [Docker](https://www.docker.com/).


## Get files
Pull this repository to get all necessary Files.<br>
<code>git clone https://github.com/Reti97/mlops-docker destination-folder</code>

## Docker Setup

### Get Container from Docker Hub
Get the dockerfile with the following command from the docker hub:
<code>docker pull reti97/new-python-app:latest</code>

### Run Training for 3 Epochs
Then you can train the Model with the best parameters found in Project 1:
<code>docker run -v Path/to/your/file/main.py:/code dockerimage python main.py --learning_rate 0.00011633938221625261 --adam_epsilon 6.73493036944769e-08 --warmup_steps 23</code>

### Build Dockerfile

You can also build the dockerfile yourself with:

<code>docker build -t docker-name .</code>

## Training

The parameters are the following:
<ul>
  <li>learning_rate = 0.00011633938221625261</li>
  <li>adam_epsilon = 6.73493036944769e-08</li>
  <li>warmup_steps = 23</li>
</ul>

The rest of the parameters will be set with a default.