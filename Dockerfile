FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy source code and sample data
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the dashboard offline
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
