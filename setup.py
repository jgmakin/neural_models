import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="neural_models",
    version="0.1.0",
    author="J.G. Makin",
    author_email="jgmakin@gmail.com",
    description="probabilistic population codes and the like in numpy, tensorflow",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/jgmakin/neural_models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'utils_jgm'
    ],
)
