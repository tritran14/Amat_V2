FROM python:3.8-slim-buster

WORKDIR /python-docker

RUN apt-get update
RUN apt-get install -y tk
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "app.py"]