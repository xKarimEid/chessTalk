# Use a minimal Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
COPY main.py .
COPY tokenizers /app/tokenizers

# Install Uvicorn and Gunicorn
RUN pip install -r requirements.txt

# Expose the port on which your FastAPI application will run
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
