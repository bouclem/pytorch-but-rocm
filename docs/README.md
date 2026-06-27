# PyTorch But ROCm

A fork of PyTorch focused on AMD ROCm support, with personal additions, optimizations, and fixes.

## What is this?

PyTorch But ROCm is a downstream fork of [PyTorch](https://github.com/pytorch/pytorch) that:

- Prioritizes and targets **AMD ROCm** as the primary GPU backend
- Includes custom optimizations and fixes not yet upstreamed
- Keeps parity with upstream PyTorch while adding ROCm-specific improvements
- Aims for clean, maintainable code over feature bloat

## Project Status

**Pre-1.0.0** - Currently undergoing major cleanup before the first stable release.

See [TODO.md](./TODO.md) for planned work and [CHANGELOG.md](./CHANGELOG.md) for release history.

## Building from Source

### Prerequisites

- Python 3.10+
- C++20 compiler (clang or gcc 11.3.0+ on Linux)
- [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0+
- At least 10 GB free disk space
- 30-60 minutes for initial build

### ROCm Build (Linux)

```bash
# Clone the repository
git clone <repo-url> pytorch-but-rocm
cd pytorch-but-rocm
git submodule sync
git submodule update --init --recursive

# Install dependencies
pip install --group dev

# Run the AMD build script first
python tools/amd_build/build_amd.py

# Build
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

### Environment Variables

| Variable | Description |
| --- | --- |
| `USE_ROCM=0` | Disable ROCm support |
| `ROCM_PATH` | Path to ROCm installation (default: `/opt/rocm`) |
| `PYTORCH_ROCM_ARCH` | Target AMD GPU architecture |
| `USE_CUDA=0` | Disable CUDA support |
| `USE_XPU=0` | Disable Intel GPU support |

### CPU-Only Build

```bash
python -m pip install --no-build-isolation -v -e .
```

## Differences from Upstream PyTorch

- ROCm is the primary and best-supported GPU backend
- Custom kernel optimizations for AMD GPUs
- Bug fixes and workarounds for ROCm-specific issues
- Simplified build configuration with ROCm defaults
- Major cleanup of unused/dead code (in progress)

## Documentation

- [TODO.md](./TODO.md) - Planned work and known issues
- [CHANGELOG.md](./CHANGELOG.md) - Release history
- [Sphinx docs](./source/) - API documentation (build with `make html` in `docs/`)
- [Contributing](../CONTRIBUTING.md) - Contribution guidelines

## License

BSD-style license - see [LICENSE](../LICENSE) for details.

## Acknowledgements

Based on [PyTorch](https://github.com/pytorch/pytorch) by the PyTorch team and contributors.
