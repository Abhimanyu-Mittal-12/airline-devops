# Use an official Python runtime
FROM python:3.10-slim

# Install Java
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model
COPY app.py .
COPY airline_delay_model/ ./airline_delay_model/

# BURN THE HADOOP CHECKSUM FILES
RUN find ./airline_delay_model -name "*.crc" -type f -delete

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]