# Getting Started on NVIDIA Jetson

This quickstart demonstrates how to configure the inner loop setup on an NVIDIA Jetson device.

> This solution has been tested on an [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano).

## Steps

1. Setup Jetson Nano  
Follow [this](https://developer.NVIDIA.com/embedded/learn/get-started-jetson-nano-devkit) guide to write the Jetson Nano Developer Kit SD Card Image to your Jetson Nano microSD card and setup the device.
1. Connect video camera to Jetson Nano USB port.
1. Follow [this](https://docs.docker.com/engine/install/ubuntu/) link to install Docker Compose on Jetson Nano.

```bash
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

1. Follow [this](https://help.ubuntu.com/community/SSH/OpenSSH/Keys) link to setup up `key-based SSH login` on Jetson Nano, so that we can connect to it from VS Code on laptop.
1. Git clone this repository on Jetson Nano.
1. Add Jetson Nano as a new [SSH target](https://code.visualstudio.com/docs/remote/ssh#_remember-hosts-and-advanced-settings) on VS Code:

    - Open VS Code on your laptop and click on Remote Explorer in the left-side Activity Bar.
    - Choose `SSH Targets` from the drop down in the Remote Explorer and click on the `Add New` icon.
    - Enter the SSH connection command `ssh <jetson-device-username>@<jetson-device-IP>`
    - Enter your Jetson device password
    - Select the directory where you cloned this repository
1. Right click on Jetson Nano SSH target created in the step above and connect and open on SSH host in new VS Code window.
1. Create `.env` file by creating a copy of [`.env_template`](../../.env_template) file.
1. Update the `.env` file as follows:

    - Change CAMERA_PATH to point to `"v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"` to use video camera instead of local video file
    - Set USE_TENSOR_RT=True, so that TensorRT engine is used for model inferencing.
    - Uncomment OPENBLAS_CORETYPE=ARMV8, so that we can execute OpenCV libraries on ARM64 devices.  

   ```sh
   CAMERA_PATH="v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"
   USE_TENSOR_RT=True
   OPENBLAS_CORETYPE=ARMV8
   ```

1. Update the following configurations in [`docker-compose.yaml`](../../.devcontainer/docker-compose.yml) file:

    - `CUDA_SUPPORT = cuda`
    - `runtime = nvidia`
    - `devices = /dev/video0:/dev/video0`

    ```yaml
    ---
    version: "3"
    services:
      dev_container:
        build:
          context: ..
          dockerfile: .devcontainer/Dockerfile
          args:
            VARIANT: 3.6-bullseye
            CUDA_SUPPORT: cuda
        env_file:
          - ../.env
        environment:
          DOCKER_BUILDKIT: 1
          COMPOSE_DOCKER_CLI_BUILD: 1
        volumes:
          - ..:/workspace:cached
        command: /bin/sh -c "while sleep 1000; do :; done"
        runtime: nvidia
        devices:
          - /dev/video0:/dev/video0
    ```

    > Note: Do not commit the changes to docker-compose.yaml file.

1. Open the code in VS Code Workspace inside the dev container using the command  `Remote-Containers: Open Workspace in Container` in Command Palette and choose the `full.code-workspace` file.

  > When you make changes to the dev container [`DockerFile`](.devcontainer/Dockerfile), you need to rebuild the Docker image for the dev container using the `Remote-Containers: Rebuild and Reopen in Container` command on VS Code Command Palette
