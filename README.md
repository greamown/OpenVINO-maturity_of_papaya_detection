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

```shell
sudo ./docker/run.sh
```

### CLI mode

```shell
python3 openvino_demo.py -c ./app/app.json
```

- This "-c" of argument is a configuration file path.

## Display
<details>
    <summary> Show inference result
    </summary>
      <div align="center">
        <img width="80%" height="80%" src="./descrit/papaya_detection.gif">
      </div>

</details>