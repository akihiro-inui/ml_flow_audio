ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := mlflow_audio
TRAINING_PATTERN := training_pattern
TRAINING_PROJECT := gtzan
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: docker_build
docker_build:
	docker build \
		-t $(BASE_IMAGE_NAME):$(TRAINING_PATTERN)_$(TRAINING_PROJECT)_$(IMAGE_VERSION) \
		-f $(DOCKERFILE) .

.PHONY: train
train:
	mlflow run . --no-conda

.PHONY: ui
ui:
	mlflow ui
