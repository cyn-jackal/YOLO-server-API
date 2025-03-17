# Start by pulling the Python image
FROM python:3.10

# Set the working directory inside the Docker container
WORKDIR /app

# Copy every content from the local file to the image
COPY . .

# Install necessary system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git

# Add build argument for GitHub token (used for accessing private repositories)
# ARG GITHUB_TOKEN

# Install dependencies from requirements.txt
# RUN pip install -r requirements.txt
# RUN pip install flask nest-asyncio pyngrok gunicorn
# RUN pip install ultralytics

# Clone the YOLO11-DSConv repository using the GitHub token for authentication
# RUN git clone https://${GITHUB_TOKEN}@github.com/cyn-jackal/YOLO11-DSConv.git
RUN git clone https://github.com/cyn-jackal/YOLO11-DSConv.git

# setting appropriate permissions using
RUN chmod -R 755 /app/YOLO11-DSConv

RUN ls /app/YOLO11-DSConv

# # Navigate into the cloned repository directory and install the package in editable mode
WORKDIR /app/YOLO11-DSConv
# RUN pip install -e . --verbose
RUN pip install .

# # Move back to the /app directory
WORKDIR /app

# Set environment variables if needed
ENV PATH=/app/.virtualenvs/bin:$PATH 

# Configure the container to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
