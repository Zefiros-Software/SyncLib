FROM phusion/baseimage:latest

RUN add-apt-repository ppa:git-core/ppa -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y git wget nano acl bzip2 g++-7 build-essential
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 60
RUN update-alternatives --config g++
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60
RUN update-alternatives --config gcc

RUN mkdir /mpich-install
WORKDIR /mpich-install
RUN wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
RUN tar -xf mpich-3.2.1.tar.gz
WORKDIR /mpich-install/mpich-3.2.1
RUN ./configure --disable-fortran
RUN make
RUN make install

WORKDIR /app

RUN useradd -m -d /home/mpich -s /bin/bash mpich

USER mpich
RUN (cd /tmp; wget https://raw.githubusercontent.com/Zefiros-Software/ZPM/features/refactor/script/install-zpm.sh && bash install-zpm.sh)
RUN echo 'export PATH="$PATH:/home/mpich/.zpm/bin:/home/mpich/.zpm/zpm-cache/tools"' >> /home/mpich/.bashrc

USER root
RUN mkdir -p /app/bin
RUN chown -R mpich /app

USER mpich