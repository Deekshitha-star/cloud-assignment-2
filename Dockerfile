FROM python:3.8-slim-buster

WORKDIR /cloud-assignment-2

ADD . /cloud-assignment-2

RUN apt-get update && \
    apt-get install openjdk-11-jdk-headless wget -y

RUN wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /opt/spark && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

ENTRYPOINT ["spark-submit", "--master", "local[*]", "--class", "com.example.Testing", "cloud-assignment-2-0.0.1.jar"]

CMD ["/data/best_model", "/data/ValidationDataset.csv"]
