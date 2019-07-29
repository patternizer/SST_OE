#!/bin/sh

# PROGRAM: run_sst_oe.sh
#
# Copyright (C) 2019 Michael Taylor, University of Reading                                 
# This code was developed for the EC project "Fidelity and Uncertainty in                 
# Climate Data Records from Earth Observations (FIDUCEO).                                 
# Grant Agreement: 638822                                                                 

# The code is distributed under terms and conditions of the MIT license:
# https://opensource.org/licenses/MIT.

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.sst_oe.sh
echo python run-derive-coeffs2.py >> run.sst_oe.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.1.log < run.sst_oe.sh




