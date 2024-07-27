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

# Expose the port
EXPOSE 8501

# Entrypoint and command
ENTRYPOINT ["streamlit", "run"]
CMD ["Hello.py"]
