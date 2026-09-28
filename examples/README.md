# Reproducible examples

The two notebooks are self-contained, executed tutorials.  Their cells load
the tabulated inputs in `data/`, construct `NBessel` or `NBesselND`, call the
public transform methods, evaluate an independent direct-quadrature result,
and make the comparison plots inline.  This keeps every important package API
call visible and editable in the notebook.

The `run_numerical_validation.py` and `run_cosmology_examples.py` scripts are
the corresponding noninteractive batch workflows used to regenerate the
checked-in files in `figures/`.  The notebooks do not import either script.

Install the package and notebook dependencies from the FastNBessel repository
root with

```shell
python -m pip install '.[examples]'
```

Then execute the tutorial notebooks from the `examples/` directory:

```shell
cd examples
python -m jupyter nbconvert --to notebook --execute --inplace numerical_validation.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace cosmological_applications.ipynb
```

To regenerate the fiducial tables, batch results, checked-in figures, and
notebook sources from the repository root, use

```shell
PYTHONPATH=src python examples/generate_fiducial_inputs.py
PYTHONPATH=src python examples/run_numerical_validation.py
PYTHONPATH=src python examples/run_cosmology_examples.py
PYTHONPATH=src python examples/make_notebooks.py
```

The fiducial inputs use a flat Lambda-CDM model with
`Omega_m=0.315`, `Omega_b=0.049`, `h=0.674`, `n_s=0.965`, and
`sigma_8=0.811`.  The transfer function is the Eisenstein--Hu no-wiggle fit,
normalized numerically to `sigma_8`; growth is evaluated at
`z=0.7`.  The galaxy and projected-matter bispectrum tables use tree-level
kernels.  The saved sources are undamped physical-model inputs.  The example
cells apply only distant endpoint windows, add explicit logarithmic-grid zero
padding where needed, and compare against calculations with enlarged
boundaries.  These inputs are transparent numerical demonstrations rather
than precision predictions for a particular survey.

`quadrature.py` belongs only to the examples.  It is not imported by the
installed `nbessel` package.
