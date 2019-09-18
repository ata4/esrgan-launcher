# Note: you may need older CUDA versions on older GPUs and/or drivers,
# 10.0 should work for Kepler/410.48 and newer
FROM anibali/pytorch:cuda-10.0

# Optionally, in case you have no Nvidia GPU and want a smaller base image:
#FROM anibali/pytorch:no-cuda

# Needed for image processing
RUN pip install opencv-python-headless
