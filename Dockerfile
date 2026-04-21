# Dockerfile unique pour les deux services (Spark ML et Streamlit)
FROM python:3.10-slim-bullseye

# Installation de Java 11 (requis par Apache Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jre-headless procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configuration de JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le port est exposé pour Streamlit (ignoré par le spark-trainer)
EXPOSE 8501
