# Use a lightweight Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your ML scripts AND the weights directly into the container
COPY . .

EXPOSE 8000

CMD ["python", "src/api.py"]