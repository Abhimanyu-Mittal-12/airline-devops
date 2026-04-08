# Use an image that has Java installed (Required for Spark)
FROM openjdk:11-jre-slim

# Install Python and Pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the API script and the saved model folder
COPY app.py .
COPY airline_delay_model/ ./airline_delay_model/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]