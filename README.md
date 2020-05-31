# SST_OE

Development code for optimal estimation (OE)-driven re-harmonisation of the measurement equations used for retrieval of SST from AVHRR IR channels

## Contents

* `run-derive-coeffs2.py` - main script to be run with Python 3.6
* `functions_derive_coeffs.py` - Modular functions

The first step is to clone the latest SST_OE code and step into the check out directory: 

    $ git clone https://github.com/patternizer/SST_OE.git
    $ cd SST_OE
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and
was tested in a conda virtual environment running a 64-bit version of Python 3.6+.

SST_OE can be run from sources directly, once the following module requirements are resolved:

* `numpy`
* `xarray`
* `matplotlib`
* `seaborn`

Run with:

    $ python3 run-derive-coeffs2.py
        
## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* Michael Taylor (https://patternizer.github.io)
