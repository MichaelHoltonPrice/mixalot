from setuptools import setup, find_packages

setup(
    name="mixalot",
    version="0.1.0",
    author="Michael Holton Price",
    author_email="MichaelHoltonPrice@gmail.com",
    description="A package for working with categorical, ordinal, and numerical data",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    #python_requires=">=3.6",
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'torch',
    ],
)