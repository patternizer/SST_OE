
#!/bin/sh

# PROGRAM: run-derive-coeffs2.sh

# Copyright (C) 2019 Michael Taylor, University of Reading                                
# This code was developed for the EC project "Fidelity and Uncertainty in                 
# Climate Data Records from Earth Observations (FIDUCEO).                                 
# Grant Agreement: 638822                                                                 

# The code is distributed under terms and conditions of the MIT license:
# https://opensource.org/licenses/MIT.

echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2006.sh
echo python run-derive-coeffs2.py MTA 2006 >> run.ma_2006.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2007.sh
echo python run-derive-coeffs2.py MTA 2007 >> run.ma_2007.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2008.sh
echo python run-derive-coeffs2.py MTA 2008 >> run.ma_2008.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2009.sh
echo python run-derive-coeffs2.py MTA 2009 >> run.ma_2009.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2010.sh
echo python run-derive-coeffs2.py MTA 2010 >> run.ma_2010.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2011.sh
echo python run-derive-coeffs2.py MTA 2011 >> run.ma_2011.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2012.sh
echo python run-derive-coeffs2.py MTA 2012 >> run.ma_2012.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2013.sh
echo python run-derive-coeffs2.py MTA 2013 >> run.ma_2013.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2014.sh
echo python run-derive-coeffs2.py MTA 2014 >> run.ma_2014.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2015.sh
echo python run-derive-coeffs2.py MTA 2015 >> run.ma_2015.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2016.sh
echo python run-derive-coeffs2.py MTA 2016 >> run.ma_2016.sh
echo '. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/bin/activate mike' > run.ma_2017.sh
echo python run-derive-coeffs2.py MTA 2017 >> run.ma_2017.sh

bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2006.log < run.ma_2006.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2007.log < run.ma_2007.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2008.log < run.ma_2008.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2009.log < run.ma_2009.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2010.log < run.ma_2010.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2011.log < run.ma_2011.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2012.log < run.ma_2012.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2013.log < run.ma_2013.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2014.log < run.ma_2014.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2015.log < run.ma_2015.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2016.log < run.ma_2016.sh
bsub -q short-serial -W24:00 -R rusage[mem=60000] -M 60000 -oo run.ma_2017.log < run.ma_2017.sh




