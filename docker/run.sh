#!/bin/bash
docker run \
    --platform linux/amd64 \
    --name hima \
    -v $PWD:/app/hima \
    -p 8888:8888 \
    -it \
    -d \
    pkuderov/hima \
    zsh