# Building the webpage

To build the webpage, install [Docker]()
and build the docker image with the following instructions

```bash
docker build -t wildfenicswebpagebuilder --build-arg="USERNAME=$(whoami)" --build-arg="UID=$(id -u)" .
```

and then build the webpage with:

```bash
docker run -ti -v $(pwd):/home/$(whoami)/shared -w /home/$(whoami)/shared --rm --shm-size=512m wildfenicswebpagebuilder
```

The webpage is then build in [\_build/html/index.html](_build/html/index.html).
