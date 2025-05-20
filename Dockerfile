# Use the official Python image
FROM python:3.11.11

# Prevents Python from writing pyc files and enables immediate stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
# EXPOSE 8501
EXPOSE 5000

# Command to run Streamlit
# CMD ["streamlit", "run", "app.py"]

# # Set the default command to run the Flask app
# CMD ["python", "app.py"]

# Run Gunicorn as the WSGI server with 4 worker processes
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

# Expose the Cloud Run port
EXPOSE 8080

# Run Gunicorn with dynamic PORT (fallback to 8080)
CMD ["/bin/sh", "-c", "exec gunicorn -w 4 -b 0.0.0.0:${PORT:-8080} app:app"]