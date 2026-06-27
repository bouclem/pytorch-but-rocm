# Welcome to the PyTorch setup.py.
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   REL_WITH_DEB_INFO
#     build with optimizations and -g (debug symbols)
#
#   USE_CUSTOM_DEBINFO="path/to/file1.cpp;path/to/file2.cpp"
#     build with debug info only for specified files
#
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
#   USE_CUDA=0
#     disables CUDA build
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files (unless CXXFLAGS is set), in contrast to the
#     default behavior of autogoo and cmake build systems.)
#
#     A specific flag that can be used is
#     -DHAS_TORCH_SHOW_DISPATCH_TRACE
#       build with dispatch trace that can be enabled with
#       TORCH_SHOW_DISPATCH_TRACE=1 at runtime.
#
#   CC
#     the C/C++ compiler to use
#
#   CMAKE_FRESH=1
#     force a fresh cmake configuration run, ignoring the existing cmake cache
#
#   CMAKE_ONLY=1
#     run cmake and stop; do not build the project
#
# Environment variables for feature toggles:
#
#   DEBUG_CUDA=1
#     if used in conjunction with DEBUG or REL_WITH_DEB_INFO, will also
#     build CUDA kernels with -lineinfo --source-in-ptx.  Note that
#     on CUDA 12 this may cause nvcc to OOM, so this is disabled by default.
#
#   USE_CUDNN=0
#     disables the cuDNN build
#
#   USE_CUSPARSELT=0
#     disables the cuSPARSELt build
#
#   USE_CUDSS=0
#     disables the cuDSS build
#
#   USE_CUFILE=0
#     disables the cuFile build
#
#   USE_FBGEMM=0
#     disables the FBGEMM build
#
#   USE_MSLK=0
#     disables the MSLK build
#
#   USE_KINETO=0
#     disables usage of libkineto library for profiling
#
#   USE_NUMPY=0
#     disables the NumPy build
#
#   BUILD_TEST=0
#     disables the test build
#
#   USE_MKLDNN=0
#     disables use of MKLDNN
#
#   USE_MKLDNN_ACL
#     enables use of Compute Library backend for MKLDNN on Arm;
#     USE_MKLDNN must be explicitly enabled.
#
#   MKLDNN_CPU_RUNTIME
#     MKL-DNN threading mode: TBB or OMP (default)
#
#   USE_STATIC_MKL
#     Prefer to link with MKL statically - Unix only
#   USE_ITT=0
#     disable use of Intel(R) VTune Profiler's ITT functionality
#
#   USE_NNPACK=0
#     disables NNPACK build
#
#   USE_DISTRIBUTED=0
#     disables distributed (c10d, gloo, mpi, etc.) build
#
#   USE_TENSORPIPE=0
#     disables distributed Tensorpipe backend build
#
#   USE_GLOO=0
#     disables distributed gloo backend build
#
#   USE_MPI=0
#     disables distributed MPI backend build
#
#   USE_SYSTEM_NCCL=0
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   USE_OPENMP=0
#     disables use of OpenMP for parallelization
#
#   USE_FLASH_ATTENTION=0
#     disables building flash attention for scaled dot product attention
#
#   USE_MEM_EFF_ATTENTION=0
#    disables building memory efficient attention for scaled dot product attention
#
#   BUILD_BINARY
#     enables the additional binaries/ build
#
#   ATEN_AVX512_256=TRUE
#     ATen AVX2 kernels can use 32 ymm registers, instead of the default 16.
#     This option can be used if AVX512 doesn't perform well on a machine.
#     The FBGEMM library also uses AVX512_256 kernels on Xeon D processors,
#     but it also has some (optimized) assembly code.
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   TORCH_CUDA_ARCH_LIST
#     specify which CUDA architectures to build for.
#     ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
#     These are not CUDA versions, instead, they specify what
#     classes of NVIDIA hardware we should generate PTX for.
#
#   TORCH_XPU_ARCH_LIST
#     specify which XPU architectures to build for.
#     ie `TORCH_XPU_ARCH_LIST="ats-m150,lnl-m"`
#
#   PYTORCH_ROCM_ARCH
#     specify which AMD GPU targets to build for.
#     ie `PYTORCH_ROCM_ARCH="gfx900;gfx906"`
#
#   ONNX_NAMESPACE
#     specify a namespace for ONNX built here rather than the hard-coded
#     one in this file; needed to build with other frameworks that share ONNX.
#
#   BLAS
#     BLAS to be used by Caffe2. Can be MKL, Eigen, ATLAS, FlexiBLAS, or OpenBLAS. If set
#     then the build will fail if the requested BLAS is not found, otherwise
#     the BLAS will be chosen based on what is found on your system.
#
#   MKL_THREADING
#     MKL threading mode: SEQ, TBB or OMP (default)
#
#   USE_ROCM_KERNEL_ASSERT=1
#     Enable kernel assert in ROCm platform
#
#   USE_LAYERNORM_FAST_RECIPROCAL
#     If set, enables the use of builtin functions for fast reciprocals (1/x) w.r.t.
#     layer normalization. Default: enabled.
#
#   USE_ROCM_CK_GEMM=1
#     Enable building CK GEMM backend in ROCm platform
#
#   USE_ROCM_CK_SDPA=1
#     Enable building CK SDPA backend in ROCm platform
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   CUDA_HOME (Linux/OS X)
#   CUDA_PATH (Windows)
#     specify where CUDA is installed; usually /usr/local/cuda or
#     /usr/local/cuda-x.y
#   CUDAHOSTCXX
#     specify a different compiler than the system one to use as the CUDA
#     host compiler for nvcc.
#
#   CUDA_NVCC_EXECUTABLE
#     Specify a NVCC to use. This is used in our CI to point to a cached nvcc
#
#   CUDNN_LIB_DIR
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARY
#     specify where cuDNN is installed
#
#   MIOPEN_LIB_DIR
#   MIOPEN_INCLUDE_DIR
#   MIOPEN_LIBRARY
#     specify where MIOpen is installed
#
#   NCCL_ROOT
#   NCCL_LIB_DIR
#   NCCL_INCLUDE_DIR
#     specify where nccl is installed
#
#   ACL_ROOT_DIR
#     specify where Compute Library is installed
#
#   LIBRARY_PATH
#   LD_LIBRARY_PATH
#     we will search for libraries in these paths
#
#   ATEN_THREADING
#     ATen parallel backend to use for intra- and inter-op parallelism
#     possible values:
#       OMP - use OpenMP for intra-op and native backend for inter-op tasks
#       NATIVE - use native thread pool for both intra- and inter-op tasks
#
#   USE_SYSTEM_LIBS (work in progress)
#      Use system-provided libraries to satisfy the build dependencies.
#      When turned on, the following cmake variables will be toggled as well:
#        USE_SYSTEM_CPUINFO=ON
#        USE_SYSTEM_SLEEF=ON
#        USE_SYSTEM_GLOO=ON
#        BUILD_CUSTOM_PROTOBUF=OFF
#        USE_SYSTEM_EIGEN_INSTALL=ON
#        USE_SYSTEM_FP16=ON
#        USE_SYSTEM_PTHREADPOOL=ON
#        USE_SYSTEM_PSIMD=ON
#        USE_SYSTEM_FXDIV=ON
#        USE_SYSTEM_BENCHMARK=ON
#        USE_SYSTEM_ONNX=ON
#        USE_SYSTEM_XNNPACK=ON
#        USE_SYSTEM_PYBIND11=ON
#        USE_SYSTEM_NCCL=ON
#        USE_SYSTEM_NVTX=ON
#
#   USE_MIMALLOC
#      Static link mimalloc into C10, and use mimalloc in alloc_cpu & alloc_free.
#      By default, It is only enabled on Windows and AArch64.
#
#   BUILD_LIBTORCH_WHL
#      Builds libtorch.so and its dependencies as a wheel
#
#   BUILD_PYTHON_ONLY
#      Builds pytorch as a wheel using libtorch.so from a separate wheel
#

from __future__ import annotations

import os
import sys


if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. "
        "Please switch to 64-bit Python.",
        file=sys.stderr,
    )
    sys.exit(-1)

import platform


# Also update `project.requires-python` in pyproject.toml when changing this
python_min_version = (3, 10, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        f"You are using Python {platform.python_version()}. "
        f"Python >={python_min_version_str} is required.",
        file=sys.stderr,
    )
    sys.exit(-1)

import importlib
import itertools
import subprocess
import sysconfig
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, IO

import setuptools.command.bdist_wheel
import setuptools.command.build_ext
import setuptools.errors
from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution


CWD = Path(__file__).absolute().parent

# Add the current directory to the Python path so that we can import `tools`.
# This is required when running this script with a PEP-517-enabled build backend.
#
# From the PEP-517 documentation: https://peps.python.org/pep-0517
#
# > When importing the module path, we do *not* look in the directory containing
# > the source tree, unless that would be on `sys.path` anyway (e.g. because it
# > is specified in `PYTHONPATH`).
#
sys.path.insert(0, str(CWD))  # this only affects the current process
# Add the current directory to PYTHONPATH so that we can import `tools` in subprocesses
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        str(CWD),
        os.getenv("PYTHONPATH", ""),
    ]
).rstrip(os.pathsep)

from tools.build_pytorch_libs import build_pytorch
from tools.clean import clean as _clean
from tools.generate_torch_version import get_torch_version
from tools.setup_helpers.cmake import CMake, CMakeValue
from tools.setup_helpers.env import IS_LINUX, IS_WINDOWS


def str2bool(value: str | None) -> bool:
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")


def _get_package_path(package_name: str) -> Path:
    from importlib.util import find_spec

    spec = find_spec(package_name)
    if spec:
        # The package might be a namespace package, so get_data may fail
        try:
            loader = spec.loader
            if loader is not None:
                file_path = loader.get_filename()  # type: ignore[attr-defined]
                return Path(file_path).parent
        except AttributeError:
            pass
    return CWD / package_name


BUILD_LIBTORCH_WHL = str2bool(os.getenv("BUILD_LIBTORCH_WHL"))
BUILD_PYTHON_ONLY = str2bool(os.getenv("BUILD_PYTHON_ONLY"))

if BUILD_PYTHON_ONLY:
    os.environ["BUILD_LIBTORCHLESS"] = "ON"
    os.environ["LIBTORCH_LIB_PATH"] = (_get_package_path("torch") / "lib").as_posix()

################################################################################
# Parameters parsed from environment
################################################################################

VERBOSE_SCRIPT = str2bool(os.getenv("VERBOSE", "1"))
RUN_BUILD_DEPS = True
# see if the user passed a quiet flag to setup.py arguments and respect
# that in our parts of the build
EMIT_BUILD_WARNING = False
RERUN_CMAKE = str2bool(os.environ.pop("CMAKE_FRESH", None)) or "--cmake" in sys.argv
CMAKE_ONLY = str2bool(os.environ.pop("CMAKE_ONLY", None)) or "--cmake-only" in sys.argv
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg in ("--cmake", "--cmake-only"):
        continue
    if arg == "rebuild" or arg == "build":
        arg = "build"  # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    if arg in ("develop", "install"):
        # CMAKE_ONLY only runs cmake and exits (via sys.exit in build_deps)
        # before setup() is called, so there's no need to go through pip.
        # Replace the command with "build" so setuptools doesn't complain
        # about an unrecognized command if we somehow reach setup().
        if CMAKE_ONLY:
            arg = "build"
        else:
            editable = arg == "develop"
            print(
                (
                    f"WARNING: Redirecting 'python setup.py {arg}' to "
                    f"'pip install {'-e ' if editable else ''}. -v --no-build-isolation',"
                    " for more info see https://github.com/pytorch/pytorch/issues/152276"
                ),
                file=sys.stderr,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    *(("-e", ".") if editable else (".",)),
                    "-v",
                    "--no-build-isolation",
                ],
                env={**os.environ},
            )
            sys.exit(result.returncode)
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False
    if arg in ["clean", "dist_info", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

if VERBOSE_SCRIPT:

    def report(
        *args: Any, file: IO[str] = sys.stderr, flush: bool = True, **kwargs: Any
    ) -> None:
        print(*args, file=file, flush=flush, **kwargs)

else:

    def report(
        *args: Any, file: IO[str] = sys.stderr, flush: bool = True, **kwargs: Any
    ) -> None:
        pass

    # Make distutils respect --quiet too
    setuptools.distutils.log.warn = report  # type: ignore[attr-defined]

# Constant known variables used throughout this file
TORCH_DIR = CWD / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
THIRD_PARTY_DIR = CWD / "third_party"

# CMAKE: full path to python library
if IS_WINDOWS:
    CMAKE_PYTHON_LIBRARY = (
        Path(sysconfig.get_config_var("prefix"))
        / "libs"
        / f"python{sysconfig.get_config_var('VERSION')}.lib"
    )
    # Fix virtualenv builds
    if not CMAKE_PYTHON_LIBRARY.exists():
        CMAKE_PYTHON_LIBRARY = (
            Path(sys.base_prefix)
            / "libs"
            / f"python{sysconfig.get_config_var('VERSION')}.lib"
        )
else:
    CMAKE_PYTHON_LIBRARY = Path(
        sysconfig.get_config_var("LIBDIR")
    ) / sysconfig.get_config_var("INSTSONAME")


################################################################################
# Version, create_version_file, and package_name
################################################################################

TORCH_PACKAGE_NAME = os.getenv("TORCH_PACKAGE_NAME", "torch")
LIBTORCH_PKG_NAME = os.getenv("LIBTORCH_PACKAGE_NAME", "torch_no_python")
if BUILD_LIBTORCH_WHL:
    TORCH_PACKAGE_NAME = LIBTORCH_PKG_NAME

TORCH_VERSION = get_torch_version()
report(f"Building wheel {TORCH_PACKAGE_NAME}-{TORCH_VERSION}")

cmake = CMake()


# all the work we need to do _before_ setup runs
def build_deps() -> None:
    report(f"-- Building version {TORCH_VERSION}")

    check_pydep("yaml", "pyyaml")
    build_pytorch(
        version=TORCH_VERSION,
        cmake_python_library=CMAKE_PYTHON_LIBRARY.as_posix(),
        build_python=not BUILD_LIBTORCH_WHL,
        rerun_cmake=RERUN_CMAKE,
        cmake_only=CMAKE_ONLY,
        cmake=cmake,
    )

    if CMAKE_ONLY:
        report(
            'Finished running cmake. Run "ccmake build" or '
            '"cmake-gui build" to adjust build options and '
            '"python -m pip install --no-build-isolation -v ." to build.'
        )
        sys.exit()


################################################################################
# Building dependent libraries
################################################################################

missing_pydep = """
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
""".strip()


def check_pydep(importname: str, module: str) -> None:
    try:
        importlib.import_module(importname)
    except ImportError as e:
        raise RuntimeError(
            missing_pydep.format(importname=importname, module=module)
        ) from e


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self) -> None:
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists
        # and we can get an accurate report on what is used and what is not.
        cmake_cache_vars = get_cmake_cache_vars()
        if cmake_cache_vars["USE_NUMPY"]:
            report("-- Building with NumPy bindings")
        else:
            report("-- NumPy not found")
        if cmake_cache_vars["USE_CUDNN"]:
            report(
                "-- Detected cuDNN at "
                f"{cmake_cache_vars['CUDNN_LIBRARY']}, "
                f"{cmake_cache_vars['CUDNN_INCLUDE_DIR']}"
            )
        else:
            report("-- Not using cuDNN")
        if cmake_cache_vars["USE_CUDA"]:
            report(f"-- Detected CUDA at {cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR']}")
        else:
            report("-- Not using CUDA")
        if cmake_cache_vars["USE_XPU"]:
            report(f"-- Detected XPU runtime at {cmake_cache_vars['SYCL_LIBRARY_DIR']}")
        else:
            report("-- Not using XPU")
        if cmake_cache_vars["USE_MKLDNN"]:
            report("-- Using MKLDNN")
            if cmake_cache_vars["USE_MKLDNN_ACL"]:
                report("-- Using Compute Library for the Arm architecture with MKLDNN")
            else:
                report(
                    "-- Not using Compute Library for the Arm architecture with MKLDNN"
                )
            if cmake_cache_vars["USE_MKLDNN_CBLAS"]:
                report("-- Using CBLAS in MKLDNN")
            else:
                report("-- Not using CBLAS in MKLDNN")
        else:
            report("-- Not using MKLDNN")
        if cmake_cache_vars["USE_NCCL"] and cmake_cache_vars["USE_SYSTEM_NCCL"]:
            report(
                "-- Using system provided NCCL library at "
                f"{cmake_cache_vars['NCCL_LIBRARIES']}, "
                f"{cmake_cache_vars['NCCL_INCLUDE_DIRS']}"
            )
        elif cmake_cache_vars["USE_NCCL"]:
            report("-- Building NCCL library")
        else:
            report("-- Not using NCCL")
        if cmake_cache_vars["USE_DISTRIBUTED"]:
            if IS_WINDOWS:
                report("-- Building without distributed package")
            else:
                report("-- Building with distributed package: ")
                report(f"  -- USE_TENSORPIPE={cmake_cache_vars['USE_TENSORPIPE']}")
                report(f"  -- USE_GLOO={cmake_cache_vars['USE_GLOO']}")
                report(f"  -- USE_MPI={cmake_cache_vars['USE_OPENMPI']}")
        else:
            report("-- Building without distributed package")
        if cmake_cache_vars["STATIC_DISPATCH_BACKEND"]:
            report(
                "-- Using static dispatch with "
                f"backend {cmake_cache_vars['STATIC_DISPATCH_BACKEND']}"
            )
        if cmake_cache_vars["USE_LIGHTWEIGHT_DISPATCH"]:
            report("-- Using lightweight dispatch")

        if cmake_cache_vars["USE_ITT"]:
            report("-- Using ITT")
        else:
            report("-- Not using ITT")

        super().run()

    def get_outputs(self) -> list[str]:
        outputs = super().get_outputs()
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report(f"setup.py::get_outputs returning {outputs}")
        return outputs


class bdist_wheel(setuptools.command.bdist_wheel.bdist_wheel):
    def write_wheelfile(self, *args: Any, **kwargs: Any) -> None:
        super().write_wheelfile(*args, **kwargs)

        if BUILD_LIBTORCH_WHL:
            if self.bdist_dir is None:
                raise AssertionError("self.bdist_dir must not be None")
            bdist_dir = Path(self.bdist_dir)
            # Remove extraneneous files in the libtorch wheel
            for file in itertools.chain(
                bdist_dir.rglob("*.a"),
                bdist_dir.rglob("*.so"),
            ):
                if (bdist_dir / file.name).is_file():
                    file.unlink()
            for file in bdist_dir.rglob("*.py"):
                file.unlink()
            # need an __init__.py file otherwise we wouldn't have a package
            (bdist_dir / "torch" / "__init__.py").touch()


class clean(Command):
    user_options: ClassVar[list[tuple[str, str | None, str]]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        _clean()


class BinaryDistribution(Distribution):
    # torch._C is built by CMake (torch/CMakeLists.txt), so setuptools has
    # no ext_modules. Force the wheel to be tagged as a binary distribution
    # so bdist_wheel uses the binary layout (only files listed via
    # package_data / data_files are packaged) rather than the purelib
    # layout, which would dump the entire build/lib/ tree -- including
    # CMake's test artifacts -- into the wheel at site-packages root.
    def has_ext_modules(self) -> bool:
        return True


def get_cmake_cache_vars() -> defaultdict[str, CMakeValue]:
    try:
        return defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist.
        # Probably running "python setup.py clean" over a clean directory.
        return defaultdict(lambda: False)


def configure_extension_build() -> tuple[
    list[Extension],  # ext_modules
    dict[str, type[Command]],  # cmdclass
    list[str],  # packages
    dict[str, list[str]],  # entry_points
    list[str],  # extra_install_requires
]:
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """

    cmake_cache_vars = get_cmake_cache_vars()

    ################################################################################
    # Configure compile flags
    ################################################################################

    extra_install_requires: list[str] = []

    # pypi cuda package that requires installation of cuda runtime, cudnn and cublas
    # should be included in all wheels uploaded to pypi
    pytorch_extra_install_requires = os.getenv("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
    if pytorch_extra_install_requires:
        report(f"pytorch_extra_install_requirements: {pytorch_extra_install_requires}")
        extra_install_requires.extend(
            map(str.strip, pytorch_extra_install_requires.split("|"))
        )

    ################################################################################
    # Declare extensions and package
    ################################################################################

    # torch._C is now built by CMake (torch/CMakeLists.txt) and installed
    # into the cmake install prefix, which maps to torch/ in the source tree.
    ext_modules: list[Extension] = []
    # packages that we want to install into site-packages and include them in wheels
    includes = ["torch", "torch.*", "torchgen", "torchgen.*"]
    # exclude folders that they look like Python packages but are not wanted in wheels
    excludes = ["tools", "tools.*", "caffe2", "caffe2.*"]
    if cmake_cache_vars["BUILD_FUNCTORCH"]:
        includes.extend(["functorch", "functorch.*"])
    else:
        excludes.extend(["functorch", "functorch.*"])
    packages = find_packages(include=includes, exclude=excludes)

    cmdclass = {
        "bdist_wheel": bdist_wheel,
        "build_ext": build_ext,
        "clean": clean,
    }

    entry_points = {
        "console_scripts": [
            "torchrun = torch.distributed.run:main",
        ],
        "torchrun.logs_specs": [
            "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
        ],
    }

    if cmake_cache_vars["USE_DISTRIBUTED"]:
        # Only enable fr_trace command if distributed is enabled
        entry_points["console_scripts"].append(
            "torchfrtrace = torch.distributed.flight_recorder.fr_trace:main",
        )
        entry_points["torch.distributed.backends"] = [
            "mpi = torch.distributed.distributed_c10d:_register_builtin_mpi_backend",
            "gloo = torch.distributed.distributed_c10d:_register_builtin_gloo_backend",
            "nccl = torch.distributed.distributed_c10d:_register_builtin_nccl_backend",
            "ucc = torch.distributed.distributed_c10d:_register_builtin_ucc_backend",
            "xccl = torch.distributed.distributed_c10d:_register_builtin_xccl_backend",
        ]
    return ext_modules, cmdclass, packages, entry_points, extra_install_requires


# post run, warnings, printed at the end to make them more visible
build_update_message = """
It is no longer necessary to use the 'build' or 'rebuild' targets

To install:
  $ python -m pip install --no-build-isolation -v .
To develop locally:
  $ python -m pip install --no-build-isolation -v -e .
To force cmake to re-generate native build files (off by default):
  $ CMAKE_FRESH=1 python -m pip install --no-build-isolation -v -e .
""".strip()


def print_box(msg: str) -> None:
    msg = textwrap.dedent(msg).strip()
    lines = ["", *msg.split("\n"), ""]
    max_width = max(len(l) for l in lines)
    print("+" + "-" * (max_width + 4) + "+", file=sys.stderr, flush=True)
    for line in lines:
        print(f"|  {line:<{max_width}s}  |", file=sys.stderr, flush=True)
    print("+" + "-" * (max_width + 4) + "+", file=sys.stderr, flush=True)


def main() -> None:
    if BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY:
        raise RuntimeError(
            "Conflict: 'BUILD_LIBTORCH_WHL' and 'BUILD_PYTHON_ONLY' can't both be 1. "
            "Set one to 0 and rerun."
        )

    install_requires = [
        "filelock",
        "typing-extensions>=4.10.0",
        "setuptools>=77.0.3",
        "sympy>=1.13.3",
        "networkx>=2.5.1",
        "jinja2",
        "fsspec>=0.8.5",
    ]
    if BUILD_PYTHON_ONLY:
        install_requires += [f"{LIBTORCH_PKG_NAME}=={TORCH_VERSION}"]

    # Parse the command line and check the arguments before we proceed with
    # building deps and setup. We need to set values so `--help` works.
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.errors.BaseError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if RUN_BUILD_DEPS:
        build_deps()

    (
        ext_modules,
        cmdclass,
        packages,
        entry_points,
        extra_install_requires,
    ) = configure_extension_build()
    install_requires += extra_install_requires

    torch_package_data = [
        "py.typed",
        # torch._C is built by CMake (torch/CMakeLists.txt) and installed into
        # torch/.  Match this interpreter's exact extension suffix, not a glob:
        # build_all.sh reuses one build/ dir across Python versions, so a glob
        # would also pick up other ABIs' _C modules left behind in the tree.
        f"_C{sysconfig.get_config_var('EXT_SUFFIX')}",
        "bin/*",
        "bin/**/*",
        "test/*",
        "*.pyi",
        "**/*.pyi",
        "lib/*.pdb",
        "lib/**/*.pdb",
        "lib/*shm*",
        "lib/torch_shm_manager",
        "lib/*.h",
        "lib/**/*.h",
        "include/*.h",
        "include/**/*.h",
        "include/*.hpp",
        "include/**/*.hpp",
        "include/*.cuh",
        "include/**/*.cuh",
        "csrc/inductor/aoti_runtime/model.h",
        "_inductor/codegen/*.h",
        "_inductor/codegen/aoti_runtime/*.h",
        "_inductor/codegen/aoti_runtime/*.cpp",
        "_inductor/script.ld",
        "_inductor/kernel/flex/templates/*.jinja",
        "_inductor/kernel/templates/*.jinja",
        "_export/serde/*.yaml",
        "_export/serde/*.thrift",
        "share/cmake/ATen/*.cmake",
        "share/cmake/Caffe2/*.cmake",
        "share/cmake/Caffe2/public/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake",
        "share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake",
        "share/cmake/Gloo/*.cmake",
        "share/cmake/Tensorpipe/*.cmake",
        "share/cmake/Torch/*.cmake",
        "utils/benchmark/utils/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.cpp",
        "utils/benchmark/utils/valgrind_wrapper/*.h",
        "utils/model_dump/skeleton.html",
        "utils/model_dump/code.js",
        "utils/model_dump/*.mjs",
        "_dynamo/graph_break_registry.json",
        "tools/dynamo/gb_id_mapping.py",
    ]

    if not BUILD_LIBTORCH_WHL:
        torch_package_data += [
            "lib/libtorch_python.so",
            "lib/libtorch_python.dylib",
            "lib/libtorch_python.dll",
        ]
    if not BUILD_PYTHON_ONLY:
        torch_package_data += [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
        # XXX: Why not use wildcards ["lib/aotriton.images/*", "lib/aotriton.images/**/*"] here?
        aotriton_image_path = TORCH_DIR / "lib" / "aotriton.images"
        aks2_files = [
            file.relative_to(TORCH_DIR).as_posix()
            for file in aotriton_image_path.rglob("*")
            if file.is_file()
        ]
        torch_package_data += aks2_files
    if get_cmake_cache_vars()["USE_TENSORPIPE"]:
        torch_package_data += [
            "include/tensorpipe/*.h",
            "include/tensorpipe/**/*.h",
        ]
    if get_cmake_cache_vars()["USE_KINETO"]:
        torch_package_data += [
            "include/kineto/*.h",
            "include/kineto/**/*.h",
        ]
    torchgen_package_data = [
        "packaged/*",
        "packaged/**/*",
    ]
    package_data = {
        "torch": torch_package_data,
    }
    # some win libraries are excluded
    # these are statically linked
    exclude_windows_libs = [
        "lib/dnnl.lib",
        "lib/kineto.lib",
        "lib/libprotobuf-lite.lib",
        "lib/libprotobuf.lib",
        "lib/libprotoc.lib",
    ]
    exclude_package_data = {
        "torch": exclude_windows_libs,
    }

    if not BUILD_LIBTORCH_WHL:
        package_data["torchgen"] = torchgen_package_data
        exclude_package_data["torchgen"] = ["*.py[co]"]
    else:
        # no extensions in BUILD_LIBTORCH_WHL mode
        ext_modules = []

    setup(
        name=TORCH_PACKAGE_NAME,
        version=TORCH_VERSION,
        distclass=BinaryDistribution,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=install_requires,
        package_data=package_data,
        exclude_package_data=exclude_package_data,
        # Disable automatic inclusion of data files because we want to
        # explicitly control with `package_data` above.
        include_package_data=False,
    )
    if EMIT_BUILD_WARNING:
        print_box(build_update_message)


if __name__ == "__main__":
    main()
