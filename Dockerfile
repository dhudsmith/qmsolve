FROM python:3.9-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install requirements
WORKDIR /home
COPY requirements.txt /home
RUN pip3 install -r requirements.txt --no-cache-dir

# expose port for pycharm
EXPOSE 8000