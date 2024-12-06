FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/xdit-project/xDiT.git /tmp/xDiT
RUN pip install -e /tmp/xDiT

