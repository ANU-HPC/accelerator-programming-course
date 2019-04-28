#!/usr/bin/env bash
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
touch $XSOCK

xhost local:root

docker build -t workspace .
docker run \
    --tty \
    --network=host \
    --env DISPLAY=unix$DISPLAY \
    --volume $XAUTH:/root/.Xauthority \
    --volume $XSOCK:/tmp/.X11-unix  \
    --privileged \
    --runtime=nvidia \
    --mount src=`pwd`,target=/workspace,type=bind \
    -it workspace /bin/bash

