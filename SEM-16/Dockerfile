FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN conda env create -n myenv --file=requirements.yml

ENTRYPOINT /opt/conda/bin/conda run -n myenv python main.py $INPUT_TEST $OUTPUT_TEST

# ENTRYPOINT ["tail", "-f", "/dev/null"]
