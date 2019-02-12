FROM tensorflow/tensorflow:1.12.0-gpu-py3
LABEL maintainer="public@mmakowski.com"
 
ENV LANG C.UTF-8

# Additional Debian packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install git \
  graphviz \
  openjdk-8-jdk-headless \
  maven

# Additional Python packages
RUN pip install keras==2.2.0 \
  git+https://github.com/hyperopt/hyperopt.git@1785e4d5280695b6b1e83b21ccf9c32d6999d3a3 \
  tqdm==4.19.4 \
  pydot==1.2.4 \
  ggplot==0.11.5 \
  gitpython==2.1.11 \
  mypy==0.620 \
  seaborn==0.9.0

# Configure tqdm
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

WORKDIR "/mnt"
