FROM phusion/baseimage:latest

RUN add-apt-repository ppa:git-core/ppa -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y git wget nano acl bzip2 g++-7 build-essential
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 60
RUN update-alternatives --config g++

RUN mkdir /openmpi-install
WORKDIR /openmpi-install
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2
RUN tar -xf openmpi-3.0.0.tar.bz2
WORKDIR /openmpi-install/openmpi-3.0.0
RUN ./configure
RUN make
RUN make install

WORKDIR /app
RUN rm -rf /openmpi
# Configure for extra slots on localhost
RUN echo 'localhost slots=8' >> /usr/local/etc/openmpi-default-hostfile

RUN useradd -m -d /home/openmpi -s /bin/bash openmpi

USER openmpi
RUN (cd /tmp; wget https://raw.githubusercontent.com/Zefiros-Software/ZPM/features/refactor/script/install-zpm.sh && bash install-zpm.sh)
RUN echo 'export PATH="$PATH:/home/openmpi/.zpm/bin:/home/openmpi/.zpm/zpm-cache/tools"' >> /home/openmpi/.bashrc
RUN echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"' >> /home/openmpi/.bashrc
RUN echo 'export OMPI_MCA_mpi_yield_when_idle=1' >> /home/openmpi/.bashrc

USER root
RUN mkdir -p /app/bin
RUN chown -R openmpi /app

USER openmpi