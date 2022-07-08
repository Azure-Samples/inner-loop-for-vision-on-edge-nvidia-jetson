"""This module is used to provide the local text detection engine with CUDA and TensorRT and it can run on ARM."""
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from src.edgeinferencing.config import EdgeModelConfig
from typing import List

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class Engine:
    """
    Engine class using TensorRT and CUDA
    """

    def __init__(self, config: EdgeModelConfig) -> None:
        """
        Initialize the engine

        @param
            logger (Logger): logger for the engine
        """
        self._original_model_path = config.original_model_path
        self._engine_path = config.engine_path
        self._profile_config = config.profile_config
        self._trt_engine = None
        self._trt_context = None

        self.ctx = None

    def _build_engine(self):
        # https://github.com/NVIDIA/TensorRT/blob/main/samples/python/engine_refit_onnx_bidaf/build_and_refit_engine.py

        builder = trt.Builder(TRT_LOGGER)
        builder.max_batch_size = 1
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)

        print("Loading ONNX file from path {}...".format(self._original_model_path))
        with open(self._original_model_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")

        # network.get_input(0).shape = [10, 1]

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.max_workspace_size = 1 << 28  # 256MiB

        for profile_config in self._profile_config:
            profile = builder.create_optimization_profile()
            for layer_name, shape_list in profile_config.items():
                min_shape, opt_shape, max_shape = shape_list
                profile.set_shape(layer_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        print(
            "Building an engine from file {}; this may take a while...".format(
                self._original_model_path
            )
        )

        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        with open(self._engine_path, "wb") as f:
            f.write(plan)
        return engine

    def initialize(self):
        """
        initialize the engine and start CUDA context
        """
        cuda.init()
        device = cuda.Device(0)  # enter your Gpu id here
        self.ctx = device.make_context()

        self.get_engine()

    def get_engine(self):
        """
        get inference engine and execution context
        """
        if Path(self._engine_path).exists():
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(self._engine_path))
            with open(self._engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())

        if self._trt_engine is None:
            print(
                "WARN: Failed to load engine, creating new engine, it may take a while..."
            )
            self._trt_engine = self._build_engine()
        self._trt_context = self._trt_engine.create_execution_context()

    def inference_single(
        self, input_data: np.ndarray, profile_idx: int = 0, profiling: bool = False
    ) -> List:
        """
        perform inference on the given input data
        note the input data is not a list

        @param
            input_data (np.ndarray): input data to be predict on (after preprocessing)
            profile_idx (int): index of the profile configuration
            profiling (bool): whether to enable profiling
        @return
            List: list of prediction output data
        """
        input_shape = input_data.shape
        dim_mul = input_shape[-1] * input_shape[-2]
        if not self._trt_engine:
            self.get_engine()
        # with self._trt_engine.create_execution_context() as trt_context:
        inputs = []
        outputs = []
        bindings = []
        input_binding_idx = 0
        if profiling:
            profiler = trt.tensorrt.Profiler()
            self._trt_context.profiler = profiler
        for idx, binding in enumerate(self._trt_engine):
            size = abs(trt.volume(self._trt_engine.get_binding_shape(binding)))
            size *= self._trt_engine.max_batch_size * dim_mul
            dtype = trt.nptype(self._trt_engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._trt_engine.binding_is_input(binding):
                input_binding_idx = idx
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        stream = cuda.Stream()
        self._trt_context.set_optimization_profile_async(profile_idx, stream.handle)
        self._trt_context.set_binding_shape(input_binding_idx, input_shape)
        inputs[0].host = np.ascontiguousarray(input_data)
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # # Run inference
        self._trt_context.execute_async_v2(
            bindings=bindings, stream_handle=stream.handle
        )
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        output_buffer = [out.host for out in outputs]
        return output_buffer


class HostDeviceMem(object):
    """
    Helper class for allocating host and device memory.
    """

    def __init__(self, host_mem, device_mem) -> None:
        """
        Initialize the HostDeviceMem

        @param
            host_mem (np.ndarray): host memory
            device_mem (cuda.DeviceAllocation): device memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self) -> str:
        """
        String representation of the HostDeviceMem

        @return
            str: string representation of the HostDeviceMem
        """
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self) -> str:
        """
        String representation of the HostDeviceMem

        @return
            str: string representation of the HostDeviceMem
        """
        return self.__str__()
