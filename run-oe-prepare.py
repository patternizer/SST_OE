#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import optparse
from  optparse import OptionParser
import datetime
import numpy as np
import xarray as xr
import pickle
import dask
from functions_derive_coeffs import *

def prepare_matchups(sensor,year):

    FUNCpath = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'
    import sys
    sys.path.append(FUNCpath)    

    path = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/DATA/'
    dirS = 'source/' + sensor + '/' + str(year) + '/'
    dirG = 'gbcsout/' + sensor + '/' + str(year) + '/'
    pathout = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'

    try:
        os.mkdir(pathout+dirG)
        os.mkdir(pathout+dirS)
    except:
        print('directories exist')

    fn = check_compatible_files(path, dirS, dirG)
    for f in fn:
    
        dsS = read_files(path, dirS, [f], reduce = True, dimstr = 'matchup_count')
        dsG = read_files(path, dirG, [f], reduce = True)
        keep = filter_matches(dsG, dsS, sstminQL = 5)
        nm = np.sum(keep)
        dsG['keep'] = ('record', keep)
        dsS['keep'] = ('matchup_count', keep)    
        dsGr = dsG.where(dsG.keep, drop = True)
        dsSr = dsS.where(dsS.keep, drop = True)
        dsGr.to_netcdf(path = pathout+dirG+f)
        dsSr.to_netcdf(path = pathout+dirS+f)

    return

if __name__ == "__main__":

    parser = OptionParser("usage: %prog sensor year")
    (options, args) = parser.parse_args()
    sensor = args[0]
    year = int(args[1])

    prepare_matchups(sensor,year)

