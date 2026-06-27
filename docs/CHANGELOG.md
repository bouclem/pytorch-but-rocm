# Changelog - PyTorch But ROCm

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project fork from PyTorch
- Project documentation (README, TODO, CHANGELOG)

### Changed
- Project renamed to "PyTorch But ROCm"
- ROCm positioned as the primary GPU backend

### Removed
- `codex_setup.sh` - OpenAI Codex setup script (upstream-only)
- `tools/nightly.py` - Upstream nightly checkout/install manager (1217 lines)
- `tools/nightly_hotpatch.py` - Upstream nightly hotpatch tool (220 lines)
- `tools/stale_issues.py` - GitHub stale issue management (316 lines)
- `tools/nvcc_fix_deps.py` - CUDA/nvcc dependency fixer (CUDA-specific, not for ROCm)
- `tools/extract_scripts.py` - CI YAML script extraction (upstream tooling)
- `tools/create_worktree.py` - Git worktree management (upstream dev tooling)
- `tools/vscode_settings.py` - VSCode settings generator (upstream dev tooling)
- `tools/render_junit.py` - JUnit XML rendering for CI
- `scripts/lint_urls.sh` - Upstream URL linting script
- `scripts/lint_xrefs.sh` - Upstream cross-reference linting script
- AI SLOP code in `setup.py` - 4 functions + `USE_NIGHTLY` code path (~330 lines)
- Deprecated `which()` function from `tools/setup_helpers/__init__.py`
- `codex_setup.sh` reference from `MANIFEST.in`
- Unused imports in `setup.py` (`shutil`, `tempfile`, `zipfile`, `IS_DARWIN`)
- `tools/download_mnist.py` - Unused MNIST download script (no references)
- `tools/optional_submodules.py` - Unused optional submodules helper (no references)
- `tools/substitute.py` - Unused string substitution tool (no references)
- `tools/merge_compile_commands.py` - Compile commands merger + CMake target in `PostBuildSteps.cmake`
- `tools/packaging/build_wheel.py` - Unused wheel packaging script (no references)
- `tools/embed_libomp_macos.py` - macOS-only OpenMP embedding + CMake block in `PostBuildSteps.cmake`
- 6 BUCK files (`BUCK.oss`, `BUCK.bzl`) - Meta internal build system, not used by CMake
- 35 `.bzl` files across `c10/`, `tools/`, `torch/` - Meta Buck build definitions
- `tools/build_defs/` directory (15 files) - Buck build helper definitions

### Fixed
- _Nothing yet_

---

## [1.0.0] - _Not yet released_

_Target: First stable release after major cleanup is complete._

### Added
- ROCm-first build configuration
- Cleaned and audited codebase
- Simplified build system with ROCm defaults
- Updated project metadata and documentation

### Changed
- Major code cleanup across all modules
- Build system streamlined for ROCm
- Documentation updated for fork-specific workflows

### Fixed
- ROCm-specific bugs and workarounds
- Dead code and unused dependencies removed
