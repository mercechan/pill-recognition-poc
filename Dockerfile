FROM python:3.8-slim

RUN ["/bin/mkdir", "-p", "/usr/src/app/uploads"]
RUN ["/bin/mkdir", "-p", "/usr/src/app/logs"]

RUN ["/bin/chmod", "755", "/usr/src/app/uploads"]
RUN ["/bin/chmod", "755", "/usr/src/app/logs"]

RUN python -m pip install --upgrade pip setuptools wheel
#RUN python -m pip install opencv-python
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libglib2.0-0

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "./run.py" ]