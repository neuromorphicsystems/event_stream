import builtins
import distutils.core
import os
import setuptools
import shutil
import sys
dirname = os.path.dirname(os.path.realpath(__file__))

def build_ext_factory(parameters):
    import setuptools.command.build_ext
    class build_ext(setuptools.command.build_ext.build_ext):
        def finalize_options(self):
            setuptools.command.build_ext.build_ext.finalize_options(self)
            builtins.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())
    return build_ext(parameters)

with open('README.md') as file:
    long_description = file.read()

setuptools.setup(
    name='event_stream',
    version='1.3.0',
    url='https://github.com/neuromorphicsystems/event_stream',
    author='Alexandre Marcireau',
    author_email='alexandre.marcireau@gmail.com',
    description='read event stream files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['numpy'],
    install_requires=['numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    ext_modules=[
        distutils.core.Extension(
            'event_stream',
            language='c++',
            sources=[os.path.join(dirname, 'python', 'event_stream.cpp')],
            extra_compile_args=(['-std=c++11'] if sys.platform == 'linux' else ['-std=c++11','-stdlib=libc++']),
            extra_link_args=(['-std=c++11'] if sys.platform == 'linux' else ['-std=c++11','-stdlib=libc++']),
            include_dirs=[],
            libraries=[],
    ],
    cmdclass={'build_ext': build_ext_factory})
