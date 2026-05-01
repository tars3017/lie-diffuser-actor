# =====Settings=====

IMAGE           ?= openvla-oft
TAG             ?= latest
CONTAINER       ?= openvla-oft

RUNAI_IMAGE     ?= openvla-oft-runai
RUNAI_TAG       ?= latest
RUNAI_CONTAINER ?= openvla-oft-runai

# Passwords for RunAI image (override via env vars or runai/.env)
SSH_PASSWORD         ?=
TIGERVNC_PASSWORD    ?=
CODE_SERVER_PASSWORD ?=
JUPYTER_LAB_TOKEN    ?=

# Load runai/.env if it exists (for build-runai passwords)
-include runai/.env
export

.PHONY: build up down shell \
        build-runai up-runai down-runai shell-runai

# =====Base dev image=====

## Build the base development image
build:
	docker build -t $(IMAGE):$(TAG) -f docker/Dockerfile .

## Start the base container (detached)
up:
	docker compose -f docker/docker-compose.yml up -d

## Stop and remove the base container
down:
	docker compose -f docker/docker-compose.yml down

## Open a shell in the running base container
shell:
	docker exec -it $(CONTAINER) bash

# =====RunAI image (VSCode + noVNC + Jupyter + SSH)=====

## Build the RunAI image (passwords read from runai/.env or env vars)
build-runai:
	docker build \
		--build-arg SSH_PASSWORD="$(SSH_PASSWORD)" \
		--build-arg TIGERVNC_PASSWORD="$(TIGERVNC_PASSWORD)" \
		--build-arg CODE_SERVER_PASSWORD="$(CODE_SERVER_PASSWORD)" \
		--build-arg JUPYTER_LAB_TOKEN="$(JUPYTER_LAB_TOKEN)" \
		-t $(RUNAI_IMAGE):$(RUNAI_TAG) \
		-f runai/Dockerfile .

## Start the RunAI container locally (for testing)
up-runai:
	docker compose -f runai/docker-compose.yml up -d

## Stop and remove the RunAI container
down-runai:
	docker compose -f runai/docker-compose.yml down

## Open a shell in the running RunAI container
shell-runai:
	docker exec -it $(RUNAI_CONTAINER) bash
