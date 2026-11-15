"""Packaging shim: make the wheel platform-specific but interpreter-agnostic.

All metadata lives in pyproject.toml. This file exists only to override the
wheel tag, which setuptools cannot express declaratively.

The package wraps the nlolib shared library through ctypes, so it ships a
`.so`/`.dll`/`.dylib` as package data but contains no compiled CPython
extension. That combination needs a tag setuptools does not pick on its own:

  * Left alone, the distribution looks pure and is tagged `py3-none-any`. One
    such wheel per platform collides on the index -- the second upload is
    rejected as a duplicate filename, and whichever won would hand a Linux
    `.so` to Windows users.
  * Declaring extension modules (`Distribution.has_ext_modules`) tags it
    `cp3X-cp3X-<plat>`, which is platform-correct but pins each wheel to one
    CPython minor version. Since no Python C API is linked, that would mean
    building and publishing N identical wheels per platform for nothing.

So mark the install tree impure -- which selects platlib and the running
platform's tag -- and then put the interpreter and ABI tags back to the
generic `py3`/`none`. Result: one `py3-none-<plat>` wheel per platform,
installable on every supported CPython.

The platform tag is deliberately left at whatever the build machine reports
(`linux_x86_64`, `win_amd64`, `macosx_*`). `auditwheel repair` and
`delvewheel repair` rewrite it afterwards to the tag the binary has actually
earned, rather than a floor asserted here and unverified.
"""

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Report the distribution as impure.

    This has to be answered on the Distribution rather than only on
    bdist_wheel: build_py consults it while staging, and a distribution that
    looks pure at that point gets relocated into `*.data/purelib/`, where
    auditwheel and delvewheel no longer recognise the package they are meant
    to repair. Saying it here puts the package at the wheel root in platlib.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        # An impure distribution would otherwise be tagged for the running
        # interpreter. Nothing here links the Python C API, so keep the
        # generic tags and the platform one.
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
