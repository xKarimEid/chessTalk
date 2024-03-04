This repo is my own implementation of byte pair encoding thanks to Andrej Karpathy work

Todo list
    1) Implement handling of special tokens /put special tokens in the input data
    2) Add more tests
    3) Add FastApi implementation and package it into docker

Docker file does the following:

runs the train script for training model
copys .model file
runs the fastapi server enabling encoding/decoding string by making api requests to endpoint