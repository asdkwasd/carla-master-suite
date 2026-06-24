"""
==============================================================================
  TensorRT Engine — Full FP16/INT8 Inference Pipeline.

  Supports:
  ✅ PyTorch → ONNX → TensorRT conversion
  ✅ FP16 and INT8 precision
  ✅ Dynamic batching
  ✅ Standalone inference (no SB3 dependency at runtime)
  ✅ Fallback to ONNX Runtime if TensorRT unavailable
==============================================================================
"""
import os
import json
import time
import numpy as np
import torch as th
from torch import nn

from .constants import (
    IMG_HEIGHT, IMG_WIDTH, N_LIDAR_SECTORS, STATE_DIM, DEPLOY_DIR,
    TRT_PRECISION, TRT_MAX_BATCH, TRT_WORKSPACE
)

# ── Optional imports ──────────────────────────────────────────────────────────
TRT_AVAILABLE = False
ORT_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# ONNX EXPORT (Step 1 of pipeline)
# =============================================================================
class ONNXExporter:
    """Export SB3 policy to ONNX for TensorRT or ONNX Runtime."""

    @staticmethod
    def export(model, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(DEPLOY_DIR, "carla_v15.onnx")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        policy = model.policy
        policy.eval()

        # Create a wrapper that extracts features → action
        class PolicyForward(nn.Module):
            def __init__(self, sb3_policy):
                super().__init__()
                self.features_extractor = sb3_policy.features_extractor
                self.mlp_extractor = sb3_policy.mlp_extractor
                self.action_net = sb3_policy.action_net

            def forward(self, camera, lidar, state):
                obs = {"camera": camera, "lidar": lidar, "state": state}
                features = self.features_extractor(obs)
                latent_pi, _ = self.mlp_extractor(features)
                return self.action_net(latent_pi)

        wrapper = PolicyForward(policy)
        wrapper.eval()

        # Dummy inputs
        dummy_cam = th.randint(0, 255, (1, IMG_HEIGHT, IMG_WIDTH, 3),
                               dtype=th.uint8).float()
        dummy_lidar = th.rand(1, N_LIDAR_SECTORS) * 50.0
        dummy_state = th.randn(1, STATE_DIM)

        th.onnx.export(
            wrapper,
            (dummy_cam, dummy_lidar, dummy_state),
            output_path,
            input_names=["camera", "lidar", "state"],
            output_names=["action"],
            opset_version=17,
            dynamic_axes={
                "camera": {0: "batch"},
                "lidar": {0: "batch"},
                "state": {0: "batch"},
                "action": {0: "batch"},
            },
        )

        size_mb = os.path.getsize(output_path) / 1e6
        print(f"[ONNX] Exported: {output_path} ({size_mb:.1f} MB)")
        return output_path


# =============================================================================
# TENSORRT ENGINE BUILDER (Step 2)
# =============================================================================
class TensorRTBuilder:
    """Build TensorRT engine from ONNX model."""

    @staticmethod
    def build(onnx_path: str, engine_path: str = None,
              precision: str = TRT_PRECISION) -> str:
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed. "
                             "Install: pip install tensorrt")

        if engine_path is None:
            engine_path = os.path.join(DEPLOY_DIR,
                                       f"carla_v15_{precision}.engine")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT] Parse error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parsing failed")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     TRT_WORKSPACE)

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[TRT] INT8 enabled")

        # Build engine
        print("[TRT] Building engine (this takes 1-5 minutes)...")
        t0 = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(engine_path, 'wb') as f:
            f.write(serialized)

        dt = time.time() - t0
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"[TRT] Engine saved: {engine_path} "
              f"({size_mb:.1f} MB, built in {dt:.1f}s)")
        return engine_path


# =============================================================================
# TENSORRT INFERENCE ENGINE (Step 3 — Runtime)
# =============================================================================
class TensorRTInference:
    """
    Run inference on TensorRT engine.
    Zero SB3 dependency at runtime — just numpy in, numpy out.
    """

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        import pycuda.driver as cuda
        import pycuda.autoinit

        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        print(f"[TRT] Engine loaded: {engine_path}")

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = self.cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape) * np.dtype(dtype).itemsize

            device_mem = self.cuda.mem_alloc(int(size))
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "device": device_mem, "host": np.empty(shape, dtype=dtype)
                })
            else:
                self.outputs.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "device": device_mem, "host": np.empty(shape, dtype=dtype)
                })

    def predict(self, camera: np.ndarray, lidar: np.ndarray,
                state: np.ndarray) -> np.ndarray:
        """
        Run inference.
        Args:
            camera: (H, W, 3) uint8
            lidar: (N_SECTORS,) float32
            state: (STATE_DIM,) float32
        Returns:
            action: (3,) float32 [steer, throttle, brake]
        """
        # Prepare inputs with batch dim
        inputs_data = [
            camera[np.newaxis].astype(np.float32),
            lidar[np.newaxis].astype(np.float32),
            state[np.newaxis].astype(np.float32),
        ]

        # Copy to device
        for inp, data in zip(self.inputs, inputs_data):
            inp["host"] = data
            self.cuda.memcpy_htod_async(inp["device"], data, self.stream)

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output back
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"][0]  # Remove batch dim


# =============================================================================
# ONNX RUNTIME FALLBACK (if no TensorRT)
# =============================================================================
class ONNXInference:
    """
    ONNX Runtime inference with TensorRT acceleration.
    Priority: TensorRT EP → CUDA EP → CPU EP
    RTX 4080 optimized: FP16, 2GB workspace, graph optimization.
    """

    def __init__(self, onnx_path: str, use_tensorrt: bool = True):
        if not ORT_AVAILABLE:
            raise RuntimeError("Install: pip install onnxruntime-gpu")

        providers = []

        # Priority 1: TensorRT Execution Provider (best perf on RTX 4080)
        if use_tensorrt:
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 2 * (1 << 30),  # 2GB for 12GB VRAM
                'trt_fp16_enable': True,                   # FP16 on Ada Lovelace
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': os.path.join(
                    os.path.dirname(onnx_path), 'trt_cache'),
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 5,
                'trt_builder_optimization_level': 5,       # Max optimization
            }
            providers.append(('TensorrtExecutionProvider', trt_options))

        # Priority 2: CUDA EP with cuDNN optimizations
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB max (of 12GB)
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': True,
            'enable_cuda_graph': True,
        }
        providers.append(('CUDAExecutionProvider', cuda_options))
        providers.append('CPUExecutionProvider')

        # Session options for max performance
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 4
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_mem_reuse = True
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers)

        active = self.session.get_providers()
        print(f"[ORT] Model loaded: {onnx_path}")
        print(f"[ORT] Active providers: {active}")
        if 'TensorrtExecutionProvider' in active:
            print(f"[ORT] ⚡ TensorRT acceleration ACTIVE (FP16)")
        elif 'CUDAExecutionProvider' in active:
            print(f"[ORT] ✓ CUDA acceleration active")

    def predict(self, camera: np.ndarray, lidar: np.ndarray,
                state: np.ndarray) -> np.ndarray:
        result = self.session.run(None, {
            "camera": camera[np.newaxis].astype(np.float32),
            "lidar": lidar[np.newaxis].astype(np.float32),
            "state": state[np.newaxis].astype(np.float32),
        })
        return result[0][0]


# =============================================================================
# SMART LOADER — Auto-selects best available backend
# =============================================================================
def load_inference_engine(model_dir: str = DEPLOY_DIR):
    """
    Auto-load best available inference engine.
    Priority: TensorRT FP16 → TensorRT INT8 → ONNX Runtime → Error

    Returns an object with .predict(camera, lidar, state) → action
    """
    # Try TensorRT engines
    for precision in ["fp16", "int8"]:
        engine_path = os.path.join(model_dir, f"carla_v15_{precision}.engine")
        if os.path.exists(engine_path) and TRT_AVAILABLE:
            try:
                return TensorRTInference(engine_path)
            except Exception as e:
                print(f"[WARN] TRT {precision} failed: {e}")

    # Fallback: ONNX Runtime
    onnx_path = os.path.join(model_dir, "carla_v15.onnx")
    if os.path.exists(onnx_path) and ORT_AVAILABLE:
        return ONNXInference(onnx_path)

    raise FileNotFoundError(
        f"No inference engine found in {model_dir}. "
        f"Run 4_DEPLOY_MASTER.py --action export first."
    )


# =============================================================================
# FULL EXPORT PIPELINE
# =============================================================================
def full_export_pipeline(sb3_model, output_dir: str = DEPLOY_DIR):
    """
    Complete export: PyTorch → ONNX → TensorRT (FP16 + INT8).
    Returns dict of exported paths.
    """
    results = {}

    # Step 1: ONNX
    onnx_path = ONNXExporter.export(sb3_model,
                                     os.path.join(output_dir, "carla_v15.onnx"))
    results["onnx"] = onnx_path

    # Step 2: TensorRT (if available)
    if TRT_AVAILABLE:
        for prec in ["fp16", "int8"]:
            try:
                engine_path = TensorRTBuilder.build(
                    onnx_path,
                    os.path.join(output_dir, f"carla_v15_{prec}.engine"),
                    precision=prec
                )
                results[f"trt_{prec}"] = engine_path
            except Exception as e:
                print(f"[WARN] TRT {prec} build failed: {e}")
    else:
        print("[INFO] TensorRT not installed — ONNX only")

    # Step 3: Config file
    config = {
        "version": "v15.0",
        "observation": {
            "camera": {"type": "semantic_segmentation",
                       "width": IMG_WIDTH, "height": IMG_HEIGHT},
            "lidar": {"sectors": N_LIDAR_SECTORS, "max_range": 50.0},
            "state_dim": STATE_DIM,
        },
        "action": {
            "steer": [-1.0, 1.0],
            "throttle": [-1.0, 1.0],
            "brake": [0.0, 1.0],
        },
        "exports": results,
        "trt_available": TRT_AVAILABLE,
        "ort_available": ORT_AVAILABLE,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    results["config"] = config_path

    print(f"\n[PIPELINE] Export complete:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results
