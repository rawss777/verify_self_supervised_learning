docker_name=verify_ssl:latest
docker run --runtime=nvidia --rm -it --shm-size=8000m -v /home:/home -v /mnt:/mnt $docker_name
