#!/usr/bin/env bash
docker build -t workspace .
docker run --runtime=nvidia -it --mount src=`pwd`,target=/workspace,type=bind workspace
