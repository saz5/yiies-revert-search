# Use a slim Python 3.9 image as the base
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for video and image processing
# ffmpeg is required by the towhee video_decode operator
# libsm6 and libxext6 are often required for OpenCV operations in a headless environment
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir reduces the size of the final image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the Flask app will run on
EXPOSE 8025

# Command to run the application
CMD ["python", "application.py"]