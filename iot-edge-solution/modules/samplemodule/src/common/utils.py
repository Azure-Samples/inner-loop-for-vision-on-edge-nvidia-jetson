import os


def get_parent_dir_path():
    path = __file__.split("/")
    path = path[:-1]
    path = "/".join(path)
    cwd = os.getcwd()
    print(f"parent dir path: {cwd}")
    return cwd


def get_camera_path():
    value = os.environ.get("CAMERA_PATH")
    if value is None:
        raise ValueError("Config environment variable CAMERA_PATH is not set.")
    print(f"Camera path: {value}")
    return value
