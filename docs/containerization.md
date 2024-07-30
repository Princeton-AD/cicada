A docker image with all requirements preinstalled is available here: https://hub.docker.com/repository/docker/ligerlac/cicada/general. It is based on tensorflow:2.15.0-gpu and comes with cuda, so it can be run out-of-the-box on suitable nvidia GPUs. To download the image:
```
docker pull ligerlac/cicada:1.0-gpu
```

## Running on lxplus
Lxplus uses apptainer / singularity instead of docker. An unpacked image is available under `/cvmfs/unpacked.cern.ch/registry.hub.docker.com/ligerlac/cicada:1.0-gpu`.

### Running interactively
Make sure to use log into a gpu-accelerated node
```
ssh username@lxplus-gpu.cern.ch
```
Then, run the image in a container using the `--nv` flag to enable the required GPU drivers:
```
singularity run --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/ligerlac/cicada\:1.0-gpu
```

### Using the image for batch jobs
In your .sub file, add the line
```
MY.SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/ligerlac/cicada:1.0-gpu"
```
as described here: https://batchdocs.web.cern.ch/containers/singularity.html. 
More info coming soon...