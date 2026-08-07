"""Build libfsz.so into the package at install time.

The Python API is a ctypes wrapper, so the wheel needs the shared library. If a
CUDA toolkit is available the library is compiled here and installed alongside
the module, which makes ``pip install`` self-contained. If nvcc is missing the
build is skipped with a warning rather than failing: the module can still find
a library through FSZ_LIB, a repository build directory, or the system search
path, and reporting that at import time is more useful than refusing to install.

Sources come from the repository when installing from a checkout, or from the
copy vendored into the sdist by ``sdist`` below.
"""

import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "fsz", "_cuda")

# Architectures to embed. PTX for the newest entry is added as well, so GPU
# generations released after this build still run.
DEFAULT_ARCHS = ["80", "90"]


def _source_dirs():
    """Return (src, include) for a checkout, or the vendored copy in an sdist."""
    repo_src = os.path.join(os.path.dirname(HERE), "src")
    repo_inc = os.path.join(os.path.dirname(HERE), "include")
    if os.path.isdir(repo_src) and os.path.isdir(repo_inc):
        return repo_src, repo_inc
    vendored_src = os.path.join(VENDOR, "src")
    vendored_inc = os.path.join(VENDOR, "include")
    if os.path.isdir(vendored_src) and os.path.isdir(vendored_inc):
        return vendored_src, vendored_inc
    return None, None


def _find_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                 "/usr/local/cuda"):
        if root:
            candidate = os.path.join(root, "bin", "nvcc")
            if os.path.exists(candidate):
                return candidate
    return None


def _archs():
    override = os.environ.get("FSZ_CUDA_ARCHITECTURES")
    if override:
        return [a for a in override.replace(",", ";").split(";") if a]
    return list(DEFAULT_ARCHS)


def _build_library(target_dir):
    src_dir, inc_dir = _source_dirs()
    if not src_dir:
        print("FSZ: CUDA sources not found; skipping the library build.")
        return False
    nvcc = _find_nvcc()
    if not nvcc:
        print("FSZ: nvcc not found; skipping the library build. Set FSZ_LIB to "
              "an existing libfsz.so, or install a CUDA toolkit and reinstall.")
        return False

    archs = _archs()
    gencode = []
    for a in archs:
        gencode += ["-gencode", "arch=compute_%s,code=sm_%s" % (a, a)]
    gencode += ["-gencode", "arch=compute_%s,code=compute_%s" % (archs[-1], archs[-1])]

    os.makedirs(target_dir, exist_ok=True)
    out = os.path.join(target_dir, "libfsz.so")
    cmd = [nvcc, "-O3", "--std=c++17", "-shared", "-Xcompiler", "-fPIC"] + gencode + [
        os.path.join(src_dir, "fsz.cu"), "-o", out,
        "-I", inc_dir, "-I", src_dir,
    ]
    print("FSZ: building the shared library for sm_%s" % ", sm_".join(archs))
    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, OSError) as exc:
        print("FSZ: the library build failed (%s); continuing without a bundled "
              "library." % exc, file=sys.stderr)
        return False
    return True


class build_py(_build_py):
    def run(self):
        super().run()
        target = os.path.join(self.build_lib, "fsz")
        _build_library(target)


class sdist(_sdist):
    """Vendor the CUDA sources so an sdist can build without the repository."""

    def run(self):
        repo_src = os.path.join(os.path.dirname(HERE), "src")
        repo_inc = os.path.join(os.path.dirname(HERE), "include")
        if os.path.isdir(repo_src):
            for name, source in (("src", repo_src), ("include", repo_inc)):
                dest = os.path.join(VENDOR, name)
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
        super().run()


setup(
    cmdclass={"build_py": build_py, "sdist": sdist},
    package_data={"fsz": ["libfsz.so", "_cuda/src/*", "_cuda/include/fsz/*"]},
)
