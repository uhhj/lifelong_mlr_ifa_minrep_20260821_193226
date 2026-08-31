# Environment audit

| Component | Release expectation | Current | Classification |
|---|---|---|---|
| GPU | CUDA-capable NVIDIA GPU | 2 × RTX 4090 | UNKNOWN (paper GPU unpublished) |
| Driver | unpublished | 580.173.02 | UNKNOWN |
| Python | 3.8 | 3.8.20 | MATCH |
| PyTorch | 1.13.0+cu116 | 1.13.0+cu116 | MATCH |
| CUDA runtime | 11.6 | 11.6 | MATCH |
| MuJoCo | 3.2.3 | 3.2.3 | MATCH |
| robosuite | 1.4.0 | 1.4.0 | MATCH |
| Transformers | 4.26.1 | 4.26.1 | MATCH |
| NumPy | 1.22.4 | 1.22.4 | MATCH |
| LIBERO | bundled official repository snapshot | bundled at upstream `a721807` plus local path fixes | MATCH for method code; local compatibility changes documented |

No environment version is changed during V2. Hugging Face offline mode only prevents failed network metadata requests and uses the already populated cache.
