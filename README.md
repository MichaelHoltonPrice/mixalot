# Overview
mixalot is a Python package for working with datasets that contain mixture
of categorical, ordinal, and continuous variables. Most notably, it supports
fitting probability density/mass functions to any dataset.

# Install
```console
git clone https://github.com/MichaelHoltonPrice/mixalot
cd mixalot
pip install .
```

# Run tests
The tests assume that both a CPU and GPU device are available to torch.

```console
python -m unittest discover tests
```