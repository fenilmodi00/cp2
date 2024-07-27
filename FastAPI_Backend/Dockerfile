# Frontend/Dockerfile
FROM python:3.10.8-slim

# Set up the work directory
WORKDIR /app

# Copy requirements.txt first (important!)
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8080

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
