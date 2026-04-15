# Building vllm-turboquant with CUDA 12.8

## Prerequisites
- CUDA Toolkit 12.8+ (`nvcc --version` should show 12.8+)
- Python 3.10 or 3.11
- setuptools, setuptools_scm, wheel

```bash
pip install setuptools setuptools_scm wheel
```

## Build from source

```bash
cd /path/to/vllm-turboquant
CUDA_HOME=/usr/local/cuda-12.8 \
VLLM_TARGET_DEVICE=cuda \
pip install -e . --no-build-isolation
```

## Verify installation

```python
import vllm
print(vllm.__version__)  # Should show turboquant build version
```

## Usage with turboquant KV cache

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-4B-AWQ",
    quantization="awq_marlin",
    kv_cache_dtype="turboquant35",
    enforce_eager=True,
    gpu_memory_utilization=0.80,
)
```

## Supported KV cache dtypes
- `turboquant35`: 3.5 bpv, 4.6x compression (maps to turbo3)
- `turboquant25`: 2.5 bpv, 6.4x compression (maps to turbo2)

## Known fixes
- `pyproject.toml`: license field updated to `{text = "Apache-2.0"}` format for setuptools compatibility
