# Environment delta audit

| Component | Release expectation | Reconciliation environment | Classification |
| --- | --- | --- | --- |
| repository / LIBERO code | upstream `a721807...` | same source lineage plus path/instrumentation patches | MATCH |
| Python | 3.8 | 3.8.20 | MATCH |
| PyTorch | 1.13.0+cu116 | 1.13.0+cu116 | MATCH |
| Torch CUDA runtime | 11.6 | 11.6 | MATCH |
| NumPy | 1.22.4 | 1.22.4 | MATCH |
| MuJoCo | 3.2.3 | 3.2.3 | MATCH |
| robosuite | 1.4.0 | 1.4.0 | MATCH |
| robomimic | 0.2.0 | 0.2.0 | MATCH |
| gym | 0.25.2 | 0.25.2 | MATCH |
| transformers | 4.26.1 | 4.26.1 | MATCH |
| CLIP | `openai/clip-vit-base-patch16` through Transformers | same model ID and Transformers version | MATCH |
| NVIDIA GPU | not specified by release | RTX 4080 SUPER or RTX 3090, one 24 GB GPU | UNKNOWN |
| NVIDIA driver | not specified | 535.230.02 | UNKNOWN |
| host CUDA compatibility | not specified | 12.2; Torch uses bundled 11.6 runtime | MINOR DELTA |
| cuDNN | not specified | 8.3.2 | UNKNOWN |
| OS / kernel | not pinned | Ubuntu 20.04.4, Linux 5.15.0-139 | UNKNOWN |

All versions pinned in `requirements.txt` match. The GPU was reassigned by the provider between completed seed42 evaluation phases, but all runs use a single 24 GB NVIDIA GPU and identical software/configuration. There is currently no concrete version mismatch justifying a second environment or changing dependencies. A second environment-matched reproduction should only be considered if all three local seeds show a stable shared gap and a specific dynamics/rendering incompatibility is identified.
