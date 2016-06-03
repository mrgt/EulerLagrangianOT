# EulerLagrangianOT
Numerical resolution of Euler's equations for incompressible fluids using semi-discrete optimal transport.
This code goes with the article *"A Lagrangian scheme for the incompressible Euler equations using optimal transport"*,
by Thomas O. Gallouet and Quentin Mérigot.

## Installation

This code requires MongeAmpere and PyMongeAmpere, available here:

https://github.com/mrgt/MongeAmpere

https://github.com/mrgt/PyMongeAmpere

Before running any of the programs, you need to set PYTHONPATH to the right location:

``` sh
export PYTHONPATH=$PYTHONPATH:/path/to/PyMongeAmpere-build
```

## Running

On a simple example:
``` sh
mkdir results
python beltrami_square.py
``` 
