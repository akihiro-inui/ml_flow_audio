ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := mlflow_audio
TRAINING_PATTERN := training_pattern
TRAINING_DATASET := gtzan
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: train
train:
	mlflow run . --no-conda

.PHONY: ui
ui:
	mlflow ui
