# Use the official Python image
FROM python:3.11.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py"]