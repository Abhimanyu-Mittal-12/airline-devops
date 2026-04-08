# 1. Use an official, supported Python runtime as a parent image
FROM python:3.10-slim

# 2. Install Java (Required for PySpark to function)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/default-java

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code and the serialized model
COPY app.py .
COPY airline_delay_model/ ./airline_delay_model/

# 7. Expose the port for FastAPI
EXPOSE 8000

# 8. Define the command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]