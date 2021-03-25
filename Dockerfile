FROM python:3.9

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN apt update
RUN apt -y install graphviz

COPY . .

WORKDIR /usr/src/app/src
