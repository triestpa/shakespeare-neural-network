#/bin/sh

RUN apt-get install -y build-essential
RUN apt-get install -y python python-dev python-setuptools
RUN apt-get install -y python-pip python-virtualenv

RUN virtualenv /opt/venv

pip install Cython