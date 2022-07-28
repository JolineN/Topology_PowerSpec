# CMB_Topology

To run the code, do 
```python run_topology.py```.

Then, the parameters in ```parameter_files/default.py``` will be used to do whatever is specified in the run_topology.py code. Now, it is set up to make 2 realizations, calculate the diagonal power spectrum to get c_ell, and to make some covariance plots.

The code should work well for ```c_l_accuracy``` of 0.9, 0.95, and 0.99. These numbers decide how accurate the realizations will be. Lower number for ```c_l_accuracy``` will also necessarily give less power (i.e. the amplitude of the a_lm will be smaller).

First the code does a substaintial amount of pre-processing. It creates the necessary spherical harmonics and transfer functions. This takes a bit of time. But it makes the creation of a_lm realizations much faster.

It should be possible to make this faster. Even though I use a substantial amount of parallelization, it does not seem like I am able to utilize all cpu power, so I will need to look into this later.

The code supports E1 and E2 with tilts and can be non-cubic.
