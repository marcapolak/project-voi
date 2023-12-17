# Project VOI

Project VOI is a Python-based application, containerized using Docker for easy deployment and scalability.

## Getting Started

These instructions will cover usage information and how to build the Docker container for the Project VOI.

### Prerequisites

- Docker installed on your machine.
- Sufficient disk space for Docker images and containers.

### Building the Docker Container

To build the Docker image for Project VOI, run the following command from the root directory of this project:

```bash
docker build -t project-voi -f docker/Dockerfile .
```
### Running the Docker Container

Once the image is built, you can run the container using:

```bash
docker run -p 4000:80 -e MODEL_VERSION=2.0.0 -v $(pwd)/output:/app/output/results project-voi
```

# Project Slides
[To the Slides👉]([https://www.example.com](https://tome.app/abc-e486/model-development-and-evaluation-clq9v10mc0hlxo3650rl5gt0h)https://tome.app/abc-e486/model-development-and-evaluation-clq9v10mc0hlxo3650rl5gt0h)
