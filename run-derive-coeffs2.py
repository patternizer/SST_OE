#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import optparse
from  optparse import OptionParser
import datetime
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pylab as plt
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pickle
import dask
from functions_derive_coeffs import *

def derive_coeffs(sensor,year,FLAG_reduce):

    if sensor == 'MTA':
        current_sensor = 'ma'
        sat = b'MTA'  # same thing but different naming convention!
    elif sensor == 'N19':
        current_sensor = 'n19'
        sat = b'N19'  # same thing but different naming convention!
    elif sensor == 'N16':
        current_sensor = 'n16'
        sat = b'N16'  # same thing but different naming convention!
    elif sensor == 'N15':
        current_sensor = 'n15'
        sat = b'N15'  # same thing but different naming convention!

    sat2 = current_sensor
    runtag = sat2+'_'+str(year)+'_'

    FUNCpath = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'
    import sys
    sys.path.append(FUNCpath)

    # Local path names for MMD and GBCS  matchup data
    if FLAG_reduce:
        path = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/DATA/'
    else:
        path = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'
    dirS = 'source/' + sensor + '/' + str(year) + '/'
    dirG = 'gbcsout/' + sensor + '/' + str(year) + '/'
    pathout = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'
    try:
        os.mkdir(pathout+dirG)
        os.mkdir(pathout+dirS)
    except:
        print('directories exist')

    # Local path for initial harmonisation coefficients
    Hpath = '/gws/nopw/j04/fiduceo/Users/mtaylor/ensemble_sst/DATA/HARMONISATION/v0.3Bet/'

    # Local path + read in LUT for Radiance <-> BT conversion
    lutdir = '/gws/nopw/j04/fiduceo/Users/mtaylor/ensemble_sst/DATA/'
    lut = read_in_LUT(sat, lutdir = lutdir)

    if FLAG_reduce:

        # Reduce the data (only need to do this on first pass through the files)
        fn = check_compatible_files(path, dirS, dirG)
        for f in fn:

            print(f)
            # Read and concatenate a selection of files (defined by fn[selected])
            dsS = read_files(path, dirS, [f], reduce = True, dimstr = 'matchup_count', satcode=sat2)
            dsG = read_files(path, dirG, [f], reduce = True, satcode=sat2)
            # Do checks for clear sky, validity, etc and also that AATSR and AVHRR are collocated
            keep = filter_matches(dsG, dsS, sstminQL = 5, satcode=sat2)
            nm = np.sum(keep)
            dsG['keep'] = ('record', keep)
            dsS['keep'] = ('matchup_count', keep)
            dsGr = dsG.where(dsG.keep, drop = True)
            dsSr = dsS.where(dsS.keep, drop = True)
            dsGr.to_netcdf(path = pathout+dirG+f)
            dsSr.to_netcdf(path = pathout+dirS+f)
    else:
        print('Data already reduced')

    fn = check_compatible_files(path, dirS, dirG)
    dsG = xr.open_mfdataset([pathout+dirG+f for f in fn ])
    dsS = xr.open_mfdataset([pathout+dirS+f for f in fn ])

    # For the valid records to keep, extract all the variables we need in arrays
    # --------------------------------------------------------------------------
    # flags, probability of clear 
    # RTTOV outputs and input SST
    # observations as counts
    # counts of ICT
    # counts of Space view
    # temperature of ICT and Instrument, and lines averaged over for Tinst
    # four prt values
    # solar and satellite angles, latitude, longitude
    # elements
    # TCWV, wind speed, secant of SZA, drifting buoy temperature

    mflag, mpclr, \
        f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
        y1, y2, y3a, y3, y4, y5, \
        c1, c2, c3a, c3, c4, c5, \
        cict3, cict4, cict5, \
        cs3, cs4, cs5, \
        tict, tinst, nl_tinst, \
        prt1,prt2,prt3,prt4, \
        solz, satz, lat, lon, \
        time, elem, line, \
        w, U, sec, xb = extract_vars(dsG, dsS, satcode = sat2)
   
    # Some plots just to see how the data are distributed
    run_checking_plots(mpclr, f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, y1, y2, y3a, y3, y4, y5, solz, satz, lat, lon, time, elem, line, w, runtag)

    # Read in the starting point calibration coefficients
    beta, ubeta, Sbeta, pnorm = read_harm_init(Hpath,current_sensor)

    # Calculate the observation BTs using these calibration coefficients

    # first, turn Tinst into normalised Tinst
    nT = (tinst - pnorm[0])/pnorm[1]
    #if current_sensor == 'ma':
    #    nT = (tinst - 286.125823)/0.049088
    #elif current_sensor == 'n19':
    #    nT = (tinst - 287.754638)/0.117681
    #elif current_sensor == 'n16':
    #    nT = (tinst - 292.672201)/3.805704
    #elif current_sensor == 'n15':
    #    nT = (tinst - 294.758564)/2.804361

    # created lict values
    lict3, lict4, lict5 = bt2rad(tict,3,lut), bt2rad(tict,4,lut), bt2rad(tict,5,lut)

    calinfo = [c3,cs3,cict3,lict3,c4,cs4,cict4,lict4,c5,cs5,cict5,lict5,nT]

    # calculate the new "observed" BTs
    l3,t3,tb3,l4,t4,tb4,l5,t5,tb5,only2chan = calc_obs(calinfo, tict, lut, beta)

    # l3 = count2rad(c3,cs3,cict3,lict3,nT,3,beta[0:4])
    # t3 = rad2bt(l3,3,lut)
    # tb3 = dbtdL(t3,3,lut) * drad_da(c3,cs3,cict3,lict3,nT,3)

    only2chan = np.where(t3 == np.nan) # flag for when only 11 and 12 are present

    # l4 = count2rad2(c4,cs4,cict4,lict4,nT,4,beta[4:8])
    # t4 = rad2bt(l4,4,lut)
    # tb4 = dbtdL(t4,4,lut) * drad_da(c4,cs4,cict4,lict4,nT,4)

    # l5 = count2rad2(c5,cs5,cict5,lict5,nT,5,beta[8:])
    # t5 = rad2bt(l5,5,lut)
    # tb5 = dbtdL(t5,5,lut) * drad_da(c5,cs5,cict5,lict5,nT,5)

    fig,ax = plt.subplots()
    plt.plot(t3,l3,'.')
    plt.title('t3 vs. l3')
    plt.savefig(runtag+'t3_l3.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(l3,tb3[0,:],'.')
    plt.title('l3 vs. tb3')
    plt.savefig(runtag+'l3_tb3.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(t4,l4,'.')
    plt.title('t4 vs. l4')
    plt.savefig(runtag+'t4_l4.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(l4,tb4[0,:],'.')
    plt.title('l4 vs. tb4')
    plt.savefig(runtag+'l4_tb4.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(t5,l5,'.')
    plt.title('t5 vs. l5')
    plt.savefig(runtag+'t5_l5.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(l5,tb5[0,:],'.')
    plt.title('l5 vs. tb5')
    plt.savefig(runtag+'l5_tb5.png')
    plt.close('all')

    # What is the uncertainty in BT implied by the calibration coefficient uncertainty?

    sens5 = np.matrix(np.mean(tb5,axis=1))
    S5 = np.matrix(Sbeta[8:,8:])
    print(np.sqrt(sens5 @ S5 @ sens5.T))

    sens4 = np.matrix(np.mean(tb4,axis=1))
    S4 = np.matrix(Sbeta[4:8,4:8])
    print(np.sqrt(sens4 @ S4 @ sens4.T))

    sens3 = np.matrix(np.nanmean(tb3,axis=1))
    S3 = np.matrix(Sbeta[:4,:4])
    print(np.sqrt(sens3 @ S3 @ sens3.T))

    # Undertake an initial OE starting from NWP prior and see how close we get buoy truth. To be illuminating, the prior must be overall unbiased so that any biases arises from radiance biases

    xt = xb-0.17+273.15 # the reference skin temperature we assume

    # Create the data for non-bias aware OE in matrix form
    Yn, Fn, Fx, Fw, Zan, K = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, x + np.mean(xt-x), y3, y4, y5, w, solz, adj_for_x = True, fix_ch3_nan = True)
    nc = np.shape(Yn)[0]

    # Set up initial estimates of covariance matrices for a first retrieval attempt using uncertainty appropriate to prior
    SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) 
    Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn)
    xoe0 = np.array(Zr[0,:] ).squeeze()
    sens0 = np.array(Ar[0,0] ).squeeze()*100
    ux0 = np.sqrt(np.array(Sr[0,0] ).squeeze())

    print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariances ------------')
    stats0 = diagnostic_plots(runtag, xoe0, xt, solz, satz, lat, lon, time, elem, w, U, sens0)

    # Now the same, but only two channels all the time

    Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn, usechan=[False,True,True])
    xoe0s = np.array(Zr[0,:] ).squeeze()
    sens0s = np.array(Ar[0,0] ).squeeze()*100
    ux0s = np.sqrt(np.array(Sr[0,0] ).squeeze())

    print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariance\
s ------------')
    stats0s = diagnostic_plots(runtag, xoe0s, xt, solz, satz, lat, lon, time, elem, w, U, sens0s)

    # Same again, new harmonisation for comparison

    # Create the data for non-bias aware OE in matrix form
    # NOTE also changes the inputs if fix_ch3_nan is TRUE
    Yn, Fn, Fx, Fw, Zan, K = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, x+ np.mean(xt-x), t3, t4, t5, w, solz, adj_for_x = True, fix_ch3_nan = True)
    nc = np.shape(Yn)[0]

    # Set up initial estimates of covariance matrices for a first retrieval attempt using uncertainty appropriate to prior
    SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) 

    # Undertake an initial OE starting from NWP prior and see how close we get buoy truth
    Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn)
    xoe0 = np.array(Zr[0,:] ).squeeze()
    sens0 = np.array(Ar[0,0] ).squeeze()*100
    ux0 = np.sqrt(np.array(Sr[0,0] ).squeeze())

    print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariances ------------')
    stats0 = diagnostic_plots(runtag, xoe0, xt, solz, satz, lat, lon, time, elem, w, U, sens0)

    # New harmonisation using only the split window

    Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn,usechan=[False,True,True])
    xoe0s = np.array(Zr[0,:] ).squeeze()
    sens0s = np.array(Ar[0,0] ).squeeze()*100
    ux0s = np.sqrt(np.array(Sr[0,0] ).squeeze())

    print('---------- STATS and PLOTS and Ralf Harmonisation initial SPLIT WINDOW ----------\
--')
    stats0s = diagnostic_plots(runtag, xoe0s, xt, solz, satz, lat, lon, time, elem, w, U, sens0s)

    drop_day = True

    # Adjust the simulations to match the starting point from skin-adjusted buoy SST

    Y0, F0, Fx0, Fw0, Z0, K0 = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, xt, t3, t4, t5, w, solz, adj_for_x = True, drop_day = drop_day)

    cw, cx, csec = np.copy([w, x, sec])
    ngt = (solz > 90)

    if drop_day:
        cw, cx, csec =  cw[ngt], cx[ngt], sec[ngt]

    # Set up initial estimates of covariance matrices using uncertainty appropriate to buoy as prior
    SSe0, SSa0 = initial_covs(csec, cw, xatype = 'buoy', scale = 0.25) 

    # Set the coefficients we are going to optimise

    # coef_list = [True, False, False, False, True, False, False, False, True, False, False, False]  # this is doing only offset coeffciients
    coef_list = [True, True , False, False, True, True , False, False, True, True , False, False]  # this is doing only offset and emissivity 

    # Set the number of strata on which to calculate a water vapour prior bias
    divsg = 5

    # List of arrays to pass all the counts and calibration counts
    calinfo = [c3,cs3,cict3,lict3,c4,cs4,cict4,lict4,c5,cs5,cict5,lict5,nT]

    ccalinfo = np.copy(calinfo)
    if drop_day:
        cc3,ccs3,ccict3,clict3,cc4,ccs4,ccict4,clict4,cc5,ccs5,ccict5,clict5,cnT = [f[ngt] for f in calinfo]
    ccalinfo = [cc3,ccs3,ccict3,clict3,cc4,ccs4,ccict4,clict4,cc5,ccs5,ccict5,clict5,cnT]

    # The initial estimate of prior TCWV bias and uncertainty, in strata
    gamma0 = np.zeros(divsg)
    ugamma0 = np.full(divsg, 0.2) # which is about 1% of the mean TCWV to start with

    beta1, gamma1, gvals1, gc1, Sbeta1, Sgamma1 = update_beta_gamma3(runtag, F0, Fx0, Fw0, Z0, SSe0, SSa0, beta, coef_list, gamma0, cw, divsg, 500000, lut, calinfo, Sbeta*400, ugamma0,  accel = 5, extrapolate = True)
    # *X and accel are just to allow values to change more rapidly -- plots verify that it is still stable

    l3r,t3r,tb3r,l4r,t4r,tb4r,l5r,t5r,tb5r,only2chan = calc_obs(calinfo, tict, lut, beta1)

    gc1 = piecewise_model(gamma1, w, gvals1, extrapolate = True)

    # Some sanity checks
    np.nanmean(t3r-t3)
    np.sum(np.nanmean(tb3,axis=1)*(beta1-beta)[0:4])
    np.nanmean(t4r-t4)
    np.sum(np.nanmean(tb4,axis=1)*(beta1-beta)[4:8])
    np.nanmean(t5r-t5)
    np.sum(np.nanmean(tb5,axis=1)*(beta1-beta)[8:])

    # Now do a new retrieval 
    Y1, F1, Fx1, Fw1, Za1, K1 = make_matrices(f3+gc1*fw3/w, f4+gc1*fw4/w, f5+gc1*fw5/w, fx3, fx4, fx5, fw3, fw4, fw5, x, x+ np.mean(xt-x), t3r, t4r, t5r, w+gc1, solz, adj_for_x = True, fix_ch3_nan = True)
    nc = np.shape(Y1)[0]

    # Set up initial estimates of covariance matrices for a first retrieval attempt using uncertainty appropriate to prior
    SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) 

    # Undertake an initial OE starting from NWP prior and see how close we get buoy truth
    Zr, Sr, Ar = optimal_estimates(Za1, K1, SSan, SSen, Y1, F1)
    xoe1 = np.array(Zr[0,:] ).squeeze()
    sens1 = np.array(Ar[0,0] ).squeeze()*100
    ux1 = np.sqrt(np.array(Sr[0,0] ).squeeze())

    print('---------- STATS and PLOTS orrection using GBCS cal & initial covariances ------------')
    stats1 = diagnostic_plots(runtag, xoe1, xt, solz, satz, lat, lon, time, elem, w, U, sens1)

    # BT impact of change in coefficients

    c3counts = [i for i in range(740, 991, 10)]
    l3old = count2rad(c3counts,np.nanmean(cs3),np.nanmean(cict3),np.nanmean(lict3),0,3,beta[0:4])
    t3old = rad2bt(l3old,3,lut)
    l3new = count2rad(c3counts,np.nanmean(cs3),np.nanmean(cict3),np.nanmean(lict3),0,3,beta1[0:4])
    t3new = rad2bt(l3new,3,lut)

    c4counts = [i for i in range(400, 860, 10)]
    l4old = count2rad(c4counts,np.nanmean(cs4),np.nanmean(cict4),np.nanmean(lict4),0,4,beta[4:8])
    t4old = rad2bt(l4old,4,lut)
    l4new = count2rad(c4counts,np.nanmean(cs4),np.nanmean(cict4),np.nanmean(lict4),0,4,beta1[4:8])
    t4new = rad2bt(l4new,4,lut)

    c5counts = [i for i in range(380, 840, 10)]
    l5old = count2rad(c5counts,np.nanmean(cs5),np.nanmean(cict5),np.nanmean(lict5),0,5,beta[8:])
    t5old = rad2bt(l5old,5,lut)
    l5new = count2rad(c5counts,np.nanmean(cs5),np.nanmean(cict5),np.nanmean(lict5),0,5,beta1[8:])
    t5new = rad2bt(l5new,5,lut)

    fig,ax = plt.subplots()
    plt.title('BT change as function of counts')
    plt.plot(c3counts,t3new-t3old)
    plt.plot(c4counts,t4new-t4old)
    plt.plot(c5counts,t5new-t5old)
    plt.legend(['Ch3', 'Ch4', 'Ch5'])
    plt.savefig(runtag+'bt-change-with-counts-c345.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.title('BT change as function of scene temperature')
    plt.plot(t3new,t3new-t3old)
    plt.plot(t4new,t4new-t4old)
    plt.plot(t5new,t5new-t5old)
    plt.legend(['Ch3', 'Ch4', 'Ch5'])
    plt.savefig(runtag+'bt-change-with-scene-temperature-c345.png')
    plt.close('all')

    fig,ax = plt.subplots()
    plt.plot(t4r,gc1,'.')
    plt.title('TCWV correction vs. BT')
    plt.savefig(runtag+'tcwv-correction-versus-bt.png')
    plt.close('all')

    results_to_netcdf(beta1,gamma1,gvals1,gc1,Sbeta1,Sgamma1,stats1,runtag)

    return

if __name__ == "__main__":

    parser = OptionParser("usage: %prog sensor year")
    (options, args) = parser.parse_args()
    sensor = args[0]
    year = int(args[1])

    FLAG_reduce = True
    derive_coeffs(sensor,year,FLAG_reduce)

print('** END')



