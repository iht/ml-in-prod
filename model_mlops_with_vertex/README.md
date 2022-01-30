# Template for Python projects

This project contains a template for Python apps that implement training and
inference of Machine Learning models. 

With this template, you will be able to:

* Use Python with the usual libraries for data science
* Write tests for your app using the provided examples

This README.md is full of details for an easier reuse of this
template. But beware, **erase its contents and include yours before
publishing your project**.

## Dependencies

The only strong requirement is that you this template is written for
**Python 3**. You can adapt it to Python 2 with minimal changes (most
likely, changing the print statements through the code)

## Utilities and common libraries

This template imports **tensorflow**. The provided scripts will install these
dependencies if they are missing in your system.

If you need to include additional dependencies, **please add new lines to the
file `requirements.txt`**, with the package name.

You can include any package available in the PyPi.

## Directories structure

In the top dir, you will find the following two files:

- `README.md`: This file will be shown as the default page in the
  Overview of your project in Bitbucket or Stash. Use it as an example
  of a README for your project.
- `setup.py`: The main `setuptools`script for your project. You should
  edit it to change the common properties of your project. If you want
  to change the version of your project, edit `src/__init__.py` instead.
- `requirements.txt`: Dependencies that must be installed for your
  project to work. Include one package name per line. The packages
  should be available in the official PyPi repository

There is also a hidden file:

- `.gitignore`: Excludes many temporary and generated files from Git.

The template has the following folders:

- `trainer`: Directory for the sources. This directory is a Python
  package, with the following contents:
    - `__init__.py`: Edit this file **to update the version** the
      version of your project, and for any other tasks common to your
      package
    - `task.py`: A simple Python script with a main function,
      and some small functions (intended to showcase how to write a
      test).
- `tests`: This directory contains the tests included in this template
  as examples. Use the files included here as templates to write your
  own tests.
  
## A note on testing

This template is using [PyTest](http://pytest.org) for the tests, with
some additional plugins for coverage calculations.

To launch the tests, PyTest offers different options. **The recommended way to
trigger the unit tests is by running the following command**:

```shell
$ python setup.py test
```

You can also use the following options:

* The `pytest` script
* Calling it as a module in the top dir of your project (this is equivalent to
  running `python setup.py test`)

We **discourage the use of the `pytest` script for testing**. If you
use this script, the top dir of your project is not included in the
path, and you need to explicitly add it to your test. Only after that
you are able to import your own modules for the tests.

For instance, if your module is called `src`, and you are using the
`pytest` script, you will need to do something like the following:

``` python
import os
import sys
sys.path.insert(0, os.path.pardir)
from trainer import ...
```

This is a violation of the Python PEP8 style guide, because you should always 
group all the `import` statements together. But this is impossible for
your `src` module unless you add it to the path.

This problem does not exist if you use PyTest as a module. This is the
approach used in this template. In that case, the module `src` is
simply available in the path, you don't need to tweak any system
path. When using `python -m pytest`, the previous code would become:

```python
from trainer import ...
```

So for testing your project in local, please do `python -m pytest`
rather than using `pytest`. In the CI jobs, the default pipeline
scripts are using `python -m pytest`.
