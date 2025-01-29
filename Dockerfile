FROM ubuntu:jammy AS base

# Set environment variables to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies & Python 3.10
RUN apt update && apt install -y  python3 \
                                  python3-pip \
                                  python3-pyaudio \
                                  portaudio19-dev \
                                  libgl1-mesa-glx \
                                  libglib2.0-0 \
                                  python3-venv \
                                  x11-apps

RUN rm -rf /var/lib/apt/lists/* 

WORKDIR /presence

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "-u", "test_camera.py" ]