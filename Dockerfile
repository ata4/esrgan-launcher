FROM anibali/pytorch:cuda-10.0

RUN sudo apt-get update && sudo apt-get install libsm6 libxrender1 libfontconfig1 && sudo rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python
