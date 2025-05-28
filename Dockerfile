# Use a lightweight Python image
FROM python:3.11-slim

# 1) Install OS deps if needed (none here) and set workdir
WORKDIR /app

# 2) Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your service code
COPY serve/ serve/

# 4) Expose port and run with Uvicorn
EXPOSE 80
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "80"]
