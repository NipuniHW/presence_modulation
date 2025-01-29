# presence_modulation

## Install dependencies

```bash
sudo apt update && sudo apt install -y portaudio19-dev python3-pyaudio
```

## Using the dockerfile

Allow X11 connections (Run this on your host machine before starting Docker):

```bash
xhost +local:
```

Build and start the dockerfile

```bash
docker compose build
docker compose up
```