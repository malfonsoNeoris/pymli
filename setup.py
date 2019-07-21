import setuptools
from os import path


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pymli",
    version="0.0.9",
    author="Franco Lianza",
    author_email="lianza.fl@gmail.com",
    description="A Python Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flianza/pymli",
    keywords=['outlier detection', 'anomaly detection', 'machine learning', 'neural networks'],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)