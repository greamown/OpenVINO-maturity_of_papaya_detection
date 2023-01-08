# Intel-OpenVINO-The maturer detection of papayas
This is using OpenVINO source code to infer the difference maturer of papayas. We use segmentation and classification of image processing methods to run this project.

* [See What's New](#see-whats-new)
* [Pre-requirements](#pre-requirements)
* [Build docker image](#build-docker-image)
* [CLI mode](#cli-mode)

# See What's New
- Using OpenVINO to infer segmentation/classification model
- This target is difference maturer of papayas

# Getting Started

### Pre-requirements
Install docker before installing the docker container.
- [Tutorial-docker](https://docs.docker.com/engine/install/ubuntu/)

### Build docker image
```shell
sudo chmod 777 ./docker
sudo ./docker/build.sh
```
##  Run container
### CLI mode

```shell
sudo ./docker/run.sh
```

## Display
<!-- <div align="center">
  <img width="100%" height="100%" src="">
</div> -->