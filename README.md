# Overview
mixalot is a Python package for working with datasets that contain mixture
of categorical, ordinal, and continuous variables. Most notably, it supports
fitting probability density/mass functions to any dataset.

# Install in a virtual environment with Python 3.9

If necessary, install Python 3.9:
```bash
winget install Python.Python.3.9
```

```bash
git clone https://github.com/MichaelHoltonPrice/mixalot
cd mixalot
```

Make (if necessary) and activate a virtual environment:

```bash
py -3.9 -m venv venv
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate
```

Install the package:

```bash
pip install .
```

# Run tests
The tests assume that both a CPU and GPU device are available to torch.
Something like this may work to install torch for GPU usage, but consider
searching the internet for an up-to-date approach applicable to your situation:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Run all of the unit tests:

```bash
python -m pytest
```

Run just the tests in one file:

```bash
python -m pytest tests/test_models.py
```