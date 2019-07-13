import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymli",
    version="0.0.4",
    author="Franco Lianza",
    author_email="lianza.fl@gmail.com",
    description="A Python Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flianza/pymli",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)