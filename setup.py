import builtins
import distutils.core
import pathlib
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import sys


class build_ext(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False  # type: ignore
        import numpy

        self.include_dirs.append(numpy.get_include())


with open("README.md") as file:
    long_description = file.read()

extra_args = []
if sys.platform == "linux":
    extra_args += ["-std=c++11"]
elif sys.platform == "darwin":
    extra_args += ["-std=c++11", "-stdlib=libc++"]

setuptools.setup(
    name="event_stream",
    version="1.4.3",
    url="https://github.com/neuromorphicsystems/event_stream",
    author="Alexandre Marcireau",
    author_email="alexandre.marcireau@gmail.com",
    description="Read and write Event Stream (.es) files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["numpy"],
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        setuptools.extension.Extension(
            "event_stream",
            language="c++",
            sources=[str(pathlib.Path("python") / "event_stream.cpp")],
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
            include_dirs=[],
            libraries=[],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
