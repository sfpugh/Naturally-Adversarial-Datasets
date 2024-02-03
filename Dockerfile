FROM ubuntu:20.04
# Install Python 3.8 and pip3
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
# Create a symbolic link to alias python3.8 as python
RUN ln -s /usr/bin/python3.8 /usr/bin/python
# Copy the code
WORKDIR /app
COPY . /app
# Install dependencies
RUN python3.8 -m pip install --upgrade pip 
RUN python3.8 -m pip install -r requirements.txt