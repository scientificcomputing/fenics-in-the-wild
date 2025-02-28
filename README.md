# Building the webpage

To build the webpage, install [Docker](https://www.docker.com/)
and build the docker image with the following instructions

```bash
docker build -f ./docker/Dockerfile.webpage -t wildfenicswebpagebuilder .
```

and then build the webpage with:

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m wildfenicswebpagebuilder
```

docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m wildfenicswebpagebuilder

The webpage is then build in [\_build/html/index.html](_build/html/index.html).

## Local development environment

Add the entrypoint `/bin/bash` to the above command, i.e.

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm --shm-size=512m --entrypoint=/bin/bash wildfenicswebpagebuilder
```
