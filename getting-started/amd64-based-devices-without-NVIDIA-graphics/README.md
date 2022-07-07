# Getting Started on AMD64-Based-Devices without NVIDIA Graphics Card

This quickstart demonstrates how to configure the inner loop setup on AMD64-based-devices without NVIDIA graphics card.

For AMD64-based-devices without NVIDIA graphics card, we use ONNX Runtime for model inferencing.

## Steps

1. Git clone this repository on the AMD64-based-device and open in VS Code.
1. Create `.env` file by creating a copy of [`.env_template`](../../.env_template) file.
1. Verify the contents of `.env` as follows:

   - Ensure CAMERA_PATH points to local video file.
   - Ensure USE_TENSOR_RT=False, so that ONNX Runtime is used for model inferencing.
   - Ensure OPENBLAS_CORETYPE=ARMV8 is commented out.

   ```sh
   CAMERA_PATH="/workspace/iot-edge-solution/modules/samplemodule/local_data/demo_video.mkv"
   USE_TENSOR_RT=False
   ```

1. Ensure [`docker-compose.yaml`](../../.devcontainer/docker-compose.yml) file has the following configurations:

    - `CUDA_SUPPORT = nocuda`
    - `runtime = runc`

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
            CUDA_SUPPORT: nocuda
        env_file:
          - ../.env
        environment:
          DOCKER_BUILDKIT: 1
          COMPOSE_DOCKER_CLI_BUILD: 1
        volumes:
          - ..:/workspace:cached
        command: /bin/sh -c "while sleep 1000; do :; done"
        runtime: runc
        devices:
          - /dev/null:/dev/video0
    ```

    > Note: Do not commit the changes to docker-compose.yaml file.

1. (Optional) If the AMD64-based-device is a Linux machine and you want to use a USB camera, make the following updates:

    - docker-compose.yaml - `devices = /dev/video0:/dev/video0`
    - .env file - `CAMERA_PATH="v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"`
1. Open the code in VS Code Workspace inside the dev container using the command  `Remote-Containers: Open Workspace in Container` in Command Palette and choose the `full.code-workspace` file.

  > When you make changes to the dev container [`DockerFile`](.devcontainer\Dockerfile), you need to rebuild the Docker image for the dev container using the `Remote-Containers: Rebuild and Reopen in Container` command on VS Code Command Palette
