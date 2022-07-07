"""This module is used to provide the edge text detection engine with CUDA or without CUDA."""
from os import getenv

USE_TENSOR_RT = getenv("USE_TENSOR_RT", "False") == "True"
if USE_TENSOR_RT:
    from src.edgeinferencing.runtime.trtruntime.engine import Engine
else:
    from src.edgeinferencing.runtime.onnxruntime.engine import Engine

print(f"USE_TENSOR_RT: {USE_TENSOR_RT}. Imported engine file: {Engine.__module__}")
