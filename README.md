# Train distilbert-base-uncased from Docker

## Get files
Pull this repository to get all necessary Files.


## Docker Setup

### Get Container from Docker Hub
Get the dockerfile with the following command from the docker hub:<br>
<code>docker pull reti97/new-python-app:latest</code>

### Run Training for 3 Epochs
Then you can train the Model with the best parameters found in Project 1:<br>
<code>docker run -v Path/to/your/file/main.py:/code new-python-app python main.py --model_name_or_path distilbert-base-uncased --learning_rate 0.00011633938221625261 --adam_epsilon 6.73493036944769e-08 --warmup_steps 23</code>

### Build Dockerfile

You can also build the dockerfile yourself with:<br>
<code>docker build -t docker-name .</code>

## Training

The parameters are the following:
<ul>
  <li>learning_rate = 0.00011633938221625261</li>
  <li>adam_epsilon = 6.73493036944769e-08</li>
  <li>warmup_steps = 23</li>
</ul>

The rest of the parameters will be set with a default.