# Numerical approximation of McKean-Vlasov SDEs via stochastic gradient descent
Code to accompany the paper "Numerical approximation of McKean-Vlasov SDEs via stochastic gradient descent" https://doi.org/10.48550/arXiv.2310.13579.

## Overview
The repository is organized into three folders, each corresponding to one of the numerical experiments discussed in **Section 4 (Numerical Study)** of the paper.

Inside each folder you will find:

- a subfolder with the tables and plots obtained from the simulations,
- a Python file (.py) that collects the utility functions, including the main stochastic gradient descent algorithm,
- a Jupyter notebook (.ipynb) that calls these functions, runs the experiments, and produces the figures and tables reported in the paper.

For the Convolution-type MK-VSE experiment, all auxiliary functions are already contained in the notebook itself, so there is no separate Python file in that case.

## License 
This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).  
