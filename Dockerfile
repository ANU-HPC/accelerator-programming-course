# An Ubuntu environment configured for building the phd repo.
FROM nvidia/opencl
MAINTAINER Beau Johnston <beau.johnston@anu.edu.au>
 
ENV DEBIAN_FRONTEND noninteractive

ENV HOME /workspace
ENV USER student

# Install essential packages.
RUN apt-get update
RUN apt-get install --no-install-recommends -y software-properties-common \
    ocl-icd-opencl-dev \
    pkg-config \
    build-essential \
    git \
    cmake \
    vim \
    less \
    make \
    tmux \
    curl \
    zlib1g-dev \
    apt-transport-https \
    dirmngr \
    wget

# Install OpenCL Device Query tool
RUN git clone https://github.com/BeauJoh/opencl_device_query.git /opencl_device_query
WORKDIR /opencl_device_query
RUN make

# Intel CPU OpenCL
RUN apt-get update -q && apt-get install --no-install-recommends -yq alien wget clinfo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Download the Intel OpenCL CPU runtime and convert to .deb packages
RUN export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz" \
    && export TAR=$(basename ${RUNTIME_URL}) \
    && export DIR=$(basename ${RUNTIME_URL} .tgz) \
    && wget -q ${RUNTIME_URL} \
    && tar -xf ${TAR} \
    && for i in ${DIR}/rpm/*.rpm; do alien --to-deb $i; done \
    && rm -rf ${DIR} ${TAR} \
    && dpkg -i *.deb \
    && rm *.deb

RUN mkdir -p /etc/OpenCL/vendors/ \
    && echo /opt/intel/*/lib64/libintelocl.so > /etc/OpenCL/vendors/intel.icd
#Install CUDA
RUN apt-get update -q && apt-get install --no-install-recommends -yq nvidia-cuda-toolkit nvidia-visual-profiler nvidia-nsight

##Install PGI community edition compiler
#RUN apt-get install --no-install-recommends -y curl
#WORKDIR /downloads
#RUN curl --user-agent "aiwc" \
#    --referer "http://www.pgroup.com/products/community.htm" --location  \
#    "https://www.pgroup.com/support/downloader.php?file=pgi-community-linux-x64" > pgi.tar.gz 
#RUN tar -xvf pgi.tar.gz \
#    && export PGI_SILENT=true \
#    && export PGI_ACCEPT_EULA=accept \
#    && export PGI_INSTALL_DIR="${HOME}/pgi" \
#    && export PGI_INSTALL_NVIDIA=false \
#    && export PGI_INSTALL_AMD=false \
#    && export PGI_INSTALL_JAVA=false \
#    && export PGI_INSTALL_MPI=false \
#    && export PGI_MPI_GPU_SUPPORT=false \
#    && export PGI_INSTALL_MANAGED=false \
#    && /downloads/install_components/install
#ENV PATH "${PATH}:$/root/pgi/linux86-64-llvm/2018/bin"

#Add new sudo user -- for X-forwarding
RUN apt-get update && \
        apt-get -y install sudo

# Install LibSciBench
ENV LSB_SRC /libscibench-source
ENV LSB /libscibench
RUN git clone https://github.com/spcl/liblsb.git $LSB_SRC
WORKDIR $LSB_SRC
RUN ./configure --prefix=$LSB
RUN make
RUN make install

# Install python libraries
RUN apt-get update && \
        apt-get install -qqy python3 python3-pip && \
        pip3 install matplotlib pandas==0.23.4 && \
        pip3 install git+https://github.com/sushinoya/ggpy

# We can build for whale (Intel(R) Xeon(R) Gold 6134 CPU) by building with:
#	docker build --build-arg HOSTNAME=whale -t workspace .
ARG HOSTNAME
COPY ./intel_silent_install.cfg /intel_silent_install.cfg
RUN if [ "$HOSTNAME" = whale ]; then \
apt-get install -qqy lsb-core libnuma1 && \
export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15365/l_opencl_p_18.1.0.014.tgz" \
    && export TAR=$(basename ${RUNTIME_URL}) \
    && export DIR=$(basename ${RUNTIME_URL} .tgz) \
    && wget -q ${RUNTIME_URL} \
    && tar -xf ${TAR} \
    && rm -f /etc/OpenCL/vendors/intel.icd \
    && ${DIR}/install.sh --silent /intel_silent_install.cfg \
;fi

# Install the OpenCL c++ headers
RUN apt-get install -qqy opencl-headers python3-tk nvidia-cuda-gdb

ENV USERNAME student

RUN mkdir -p /etc/sudoers.d && \
        useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME && \
        adduser $USERNAME sudo && \
	mkdir -p /home/$USERNAME && \
	chown $USERNAME:$USERNAME /home/$USERNAME && \
	chmod 700 /home/$USERNAME && \
	usermod --home /home/$USERNAME $USERNAME

USER $USERNAME
ENV HISTFILE=/home/$USERNAME/.bash_history
ENV HOME=/home/$USERNAME
ENV PYTHONSTARTUP=/home/$USERNAME/.python_history
ENV LD_LIBRARY_PATH "${LSB}/lib:${LD_LIBRARYPATH}"

WORKDIR /workspace
CMD ["/bin/bash", "-c", "sudo chmod -R go+w /workspace ; /bin/bash"]

#########################################
# Do not modifiy above this line
# Add your additions below this line
#########################################


