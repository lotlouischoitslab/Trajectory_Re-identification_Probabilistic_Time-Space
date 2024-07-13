FROM nvidia/cuda:12.1.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Assuming compatibility is confirmed on the PyTorch website
  
# RUN pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
RUN pip3 install --upgrade torch torchvision torchaudio
RUN pip3 install numpy scipy matplotlib pandas scikit-learn

WORKDIR C:\Users\louis\Desktop\01Autonomous_Vehicles_Research\00Autonomous_Vehicles_Trajectory_Prediction_Analysis


