FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN apt-get update --allow-releaseinfo-change
RUN apt-get install -y libgomp1

RUN pip install --no-cache-dir -r docker/requirements.txt

EXPOSE 80

ENV MODEL_VERSION 2.0.0

CMD ["python", "src/model.py"]



