# TODO - PyTorch But ROCm

## Pre-1.0.0 Major Cleanup

### Code Cleanup
- [ ] Remove unused CUDA-specific code paths that are redundant with ROCm
- [ ] Audit and remove dead code across `torch/`, `aten/`, `c10/`
- [x] Clean up `setup.py` - remove AI-generated helper functions marked as "AI SLOP"
- [x] Remove unused upstream tooling files (nightly.py, stale_issues.py, nvcc_fix_deps.py, etc.)
- [x] Remove deprecated `which()` from `tools/setup_helpers/__init__.py`
- [x] Remove unused Python files (download_mnist.py, optional_submodules.py, substitute.py, etc.)
- [x] Remove Meta-internal BUCK/bzl build files (41 files + tools/build_defs/)
- [x] Remove macOS-only tools (embed_libomp_macos.py) + CMake references
- [ ] Audit 83 `# deprecated` markers across 33 files in `torch/`
- [ ] Audit 337 TODO/FIXME with remove/dead/unused/delete across 201 files
- [ ] Consolidate duplicate ROCm/HIP code paths
- [ ] Remove unused third-party dependencies
- [ ] Standardize naming: ROCm vs HIP vs AMD terminology

### Build System
- [ ] Simplify CMake configuration for ROCm-first builds
- [ ] Make ROCm the default GPU backend in build configuration
- [ ] Clean up `CMakeLists.txt` - remove unnecessary CUDA-only logic
- [ ] Update `pyproject.toml` with correct project metadata (name, version, description)
- [ ] Remove or archive unused Docker configurations

### Documentation
- [ ] Update Sphinx docs to reflect ROCm-first positioning
- [ ] Write ROCm-specific troubleshooting guide
- [ ] Document all custom changes vs upstream PyTorch
- [ ] Update CONTRIBUTING.md for fork-specific workflows

### Testing
- [ ] Audit existing tests for ROCm compatibility
- [ ] Add ROCm-specific test coverage
- [ ] Set up CI pipeline for ROCm builds
- [ ] Remove tests for unsupported platforms/features

### Project Infrastructure
- [ ] Set up proper versioning (version.txt with 1.0.0)
- [ ] Configure git branches and tagging strategy
- [ ] Clean up CI/CD configuration for fork-specific needs
- [ ] Remove upstream-specific tooling that doesn't apply

## Post-1.0.0

### Features
- [ ] Custom ROCm kernel optimizations
- [ ] Improved ROCm memory management
- [ ] Better integration with AMD profiling tools
- [ ] ROCm-specific quantization support

### Maintenance
- [ ] Regular upstream rebase strategy
- [ ] Long-term ROCm version support matrix
- [ ] Community contribution guidelines for ROCm fixes
