FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Matplotlib
RUN apt-get update && apt-get install -y libfreetype6-dev libpng-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Expose the port
EXPOSE 8080

# START COMMAND
# We use uvicorn to run the 'app' object inside 'server.py'
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
