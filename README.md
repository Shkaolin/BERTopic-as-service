[![build](https://github.com/Shkaolin/BERTopic-as-service/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Shkaolin/BERTopic-as-service/actions/workflows/build.yml)

# BERTopic-as-service

Using BERTopic as a service to create easily interpretable topics.

## Getting Started

### Create virtual environment

Install poetry:

```bash
pip install poetry
```

Install project dependencies:

```bash
poetry install --no-root
```

Activate environment:

```bash
poetry shell
```

### Setup pre-commit

Install pre-commit:

```bash
pip install pre-commit
```

Install the git hook scripts:

```bash
pre-commit install
```

### Set up minikube

Linux:

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-amd64
sudo install minikube-darwin-amd64 /usr/local/bin/minikube
```

macOS:

```bash
brew install minikube
```

## Service management

### Operations with containers

Build image:

```bash
make build
```

Start the service:

```bash
make up
```

Start all services except bertopic:

```bash
make up-dev
```

Stop the service:

```bash
make down
```

### Tests and checks

Run code checks:

```bash
make check-code
```

Run unit tests:

```bash
make unit
```

Run integration tests:

```bash
make test
```

### Deployment on k8s

Start the cluster:

```bash
minikube start
eval $(minikube -p minikube docker-env)
```

Deploy the service on the local k8s:

```bash
kubectl apply -f k8s/namespace/ && kubectl apply -R -f k8s/
```

Open MinIO Console:

```bash
minikube service minio-service -n bertopic --url
```

Open Swagger:

```bash
minikube service bertopic-service -n bertopic --url
```

Stop the cluster:

```bash
minikube stop
```

Delete the cluster:

```bash
minikube delete --all
```