import builtins
import pathlib
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import sys

dirname = pathlib.Path(__file__).resolve().parent


class build_ext(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False  # type: ignore
        import numpy

        self.include_dirs.append(numpy.get_include())


with open(dirname / "README.md") as file:
    long_description = file.read()

extra_args = []
if sys.platform == "linux":
    extra_args += ["-std=c++17"]
elif sys.platform == "darwin":
    extra_args += ["-std=c++17", "-stdlib=libc++"]

setuptools.setup(
    name="event_stream",
    version="1.6.1",
    url="https://github.com/neuromorphicsystems/event_stream",
    author="Alexandre Marcireau",
    author_email="alexandre.marcireau@gmail.com",
    description="Read and write Event Stream (.es) files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy>=1.24"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        setuptools.extension.Extension(
            "event_stream",
            language="c++",
            sources=[str(dirname / "python" / "event_stream.cpp")],
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
            include_dirs=[],
            libraries=[],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
