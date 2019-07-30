#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:04:19 2019

@author: chris
"""


import os
import datetime
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pylab as plt


#%%
    
################
# MT: 20190729 #
################
#FUNCpath = '/Users/chris/Projects/FIDUCEO/ReHarm/'
FUNCpath = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/'
import sys
sys.path.append(FUNCpath)
from functions_derive_coeffs import *


# ALL THE things to change if running for a different case / setup
    
current_sensor = 'm02' # The code for the AVHRR being looked at
sat = b'MTA'  # same thing but different naming convention!

# Local path names for counts matchup data

################
# MT: 20190729 #
################
#path = '/Users/chris/Projects/FIDUCEO/ReHarm/Data/'
path = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/DATA/'
dirS = 'source/'
dirG = 'gbcsout/'

# We will loop through all the files and reduce to the matches we want to keep

################
# MT: 20190729 #
################
# pathout = '/Users/chris/Projects/FIDUCEO/ReHarm/DataReduced/'
pathout = '/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe/RUN'

try:
    os.mkdir(pathout+dirG)
    os.mkdir(pathout+dirS)
except:
    print('directories exist')

# Local path for initial harmonisation coefficients

################
# MT: 20190729 #
################
# Hpath = '/Users/chris/Projects/FIDUCEO/Covariance/HARMONISATION/'
Hpath = '/gws/nopw/j04/fiduceo/Users/mtaylor/ensemble_sst/DATA/HARMONISATION/v0.3Bet'

# Setup for Radiance <-> BT directory

################
# MT: 20190729 #
################
# lutdir = '/Users/chris/Projects/FIDUCEO/MMD-Harm/'
lutdir = '/gws/nopw/j04/fiduceo/Users/mtaylor/ensemble_sst/DATA/'

# read look up table
lut = read_in_LUT(sat, lutdir = lutdir)

#%%
########## TYPICALLY CAN SKIP
# Tests on the various RT modules that I will use



rad3 = count2rad2(800,1000,870,0.33,0,3,np.array([2.67393749e-03,  2.21853418e-02,  1.15214074e-0]))
bt3 = rad2bt(rad3,3,lut)
bt2rad(bt3,3,lut), rad3

dbtdL(bt3,3,lut), 100*(rad2bt(rad3+0.005,3,lut)-rad2bt(rad3-0.005,3,lut))
drad_da2(800,1000,870,0.33,1,3)

500*(count2rad2(800,1000,870,0.33,0,3,np.array([2.67393749e-03 + 1e-3,  2.21853418e-02,  1.15214074e-0])) - \
    count2rad2(800,1000,870,0.33,0,3,np.array([2.67393749e-03 - 1e-3,  2.21853418e-02,  1.15214074e-0])))

500*(count2rad2(800,1000,870,0.33,0,3,np.array([2.67393749e-03 ,  2.21853418e-02 + 1e-3,  1.15214074e-0])) - \
    count2rad2(800,1000,870,0.33,0,3,np.array([2.67393749e-03 ,  2.21853418e-02 - 1e-3,  1.15214074e-0])))

500*(count2rad2(800,1000,870,0.33,1,3,np.array([2.67393749e-03 ,  2.21853418e-02,  1.15214074e-0+ 1e-3])) - \
    count2rad2(800,1000,870,0.33,1,3,np.array([2.67393749e-03 ,  2.21853418e-02,  1.15214074e-0- 1e-3])))


drad_da2(700.,1000,800,100.,1,4)
500000*(count2rad2(700,1000,800,100,1,4,np.array([1.19418001e+00,  7.07987014e-04,  8.21189525e-06+ 1e-6,  1.14700561e-03])) - \
    count2rad2(700,1000,800,100,1,4,np.array([1.19418001e+00,  7.07987014e-04,  8.21189525e-06- 1e-6,  1.14700561e-03])))

bt5 = rad2bt(100,5,lut)
dbtdL(bt5,5,lut), 100*(rad2bt(100+0.005,5,lut)-rad2bt(100-0.005,5,lut))


#%%
fn = check_compatible_files(path, dirS, dirG)

# Reduce the data (only need to do this on first pass through the files)

for f in fn:
    
    # Read and concatenate a selection of files (defined by fn[selected])
    dsS = read_files(path, dirS, [f], reduce = True, dimstr = 'matchup_count')
    dsG = read_files(path, dirG, [f], reduce = True)

    # Do checks for clear sky, validity, etc and also that AATSR and AVHRR are collocated
    keep = filter_matches(dsG, dsS, sstminQL = 5)

    nm = np.sum(keep)


    dsG['keep'] = ('record', keep)
    dsS['keep'] = ('matchup_count', keep)

    
    dsGr = dsG.where(dsG.keep, drop = True)
    dsSr = dsS.where(dsS.keep, drop = True)


    dsGr.to_netcdf(path = pathout+dirG+f)
    dsSr.to_netcdf(path = pathout+dirS+f)


#%%
############################################
# TYPICALLY START HERE AFTER FIRST RUN
#############################################
fn = check_compatible_files(path, dirS, dirG)
dsG = xr.open_mfdataset([pathout+dirG+f for f in fn ])
dsS = xr.open_mfdataset([pathout+dirS+f for f in fn ])

#%%

# For the valid records to keep, extract all the variables we need in arrays
    
# flags, probability of clear 
    # RTTOV outputs and input SST
    # observations as counts
    # counts of ICT
    # counts of Space view
    # temperature of ICT and Instrument, and lines averaged over for Tinst
    #four prt values
    # solar and satellite angles, latitude, longitude
    # elements
    #TCWV, wind speed, secant of SZA, drifting buoy temperature


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
    w, U, sec, xb = extract_vars(dsG, dsS)

#%%
   
# Some plots just to see how the data are distributed
run_checking_plots(mpclr, f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    y1, y2, y3a, y3, y4, y5, solz, satz, lat, lon, time, elem, line, w)

#%%

# Read in the starting point calibration coefficients
beta, ubeta, Sbeta = read_harm_init(Hpath,'m02')

#%%
# Calculate the observation BTs using these calibration coefficients

# first, turn Tinst into normalised Tinst
nT = (tinst - 286.125823)/0.049088 
# placeholder for now -- these numbers for MetopA need to be generalised to other sensors


# created lict values
lict3, lict4, lict5 = bt2rad(tict,3,lut), bt2rad(tict,4,lut), bt2rad(tict,5,lut)

# calculate the new "observed" BTs
l3 = count2rad(c3,cs3,cict3,lict3,nT,3,beta[0:4])
t3 = rad2bt(l3,3,lut)
plt.plot(t3,l3,'.')
tb3 = dbtdL(t3,3,lut) * drad_da(c3,cs3,cict3,lict3,nT,3)
plt.plot(l3,tb3[0,:],'.')

only2chan = np.where(t3 == np.nan) # flag for when only 11 and 12 are present


l4 = count2rad2(c4,cs4,cict4,lict4,nT,4,beta[4:8])
t4 = rad2bt(l4,4,lut)
plt.plot(t4,l4,'.')
tb4 = dbtdL(t4,4,lut) * drad_da(c4,cs4,cict4,lict4,nT,4)
plt.plot(l4,tb4[0,:],'.')

l5 = count2rad2(c5,cs5,cict5,lict5,nT,5,beta[8:])
t5 = rad2bt(l5,5,lut)
plt.plot(t5,l5,'.')
tb5 = dbtdL(t5,5,lut) * drad_da(c5,cs5,cict5,lict5,nT,5)
plt.plot(l5,tb5[0,:],'.')

#%%

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


#%%


# Undertake an initial OE starting from NWP prior and see how close we get buoy truth
# To be illuminating, the prior must be overall unbiased so that any biases arises
# from radiance biases

xt = xb-0.17+273.15 # the reference skin temperature we assume

# Create the data for non-bias aware OE in matrix form
Yn, Fn, Fx, Fw, Zan, K = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    x + np.mean(xt-x), y3, y4, y5, w, adj_for_x = True, fix_ch3_nan = True)
nc = np.shape(Yn)[0]


# Set up initial estimates of covariance matrices for a first retrieval attempt
SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) # using uncertainty appropriate to prior


Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn)
xoe0 = np.array(Zr[0,:] ).squeeze()
sens0 = np.array(Ar[0,0] ).squeeze()*100
ux0 = np.sqrt(np.array(Sr[0,0] ).squeeze())



# Printout stats and plots
print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariances ------------')
stats0 = diagnostic_plots(xoe0, xt, solz, satz, lat, lon, time, elem, w, U, sens0)


#%%
# Same again, new harmonisation for comparison

# Create the data for non-bias aware OE in matrix form
# NOTE also changes the inputs if fix_ch3_nan is TRUE
Yn, Fn, Fx, Fw, Zan, K = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    x+ np.mean(xt-x), t3, t4, t5, w, adj_for_x = True, fix_ch3_nan = True)
nc = np.shape(Yn)[0]


# Set up initial estimates of covariance matrices for a first retrieval attempt
SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) # using uncertainty appropriate to prior

# Undertake an initial OE starting from NWP prior and see how close we get buoy truth
Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn)
xoe0 = np.array(Zr[0,:] ).squeeze()
sens0 = np.array(Ar[0,0] ).squeeze()*100
ux0 = np.sqrt(np.array(Sr[0,0] ).squeeze())

# Printout stats and plots
print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariances ------------')
stats0 = diagnostic_plots(xoe0, xt, solz, satz, lat, lon, time, elem, w, U, sens0)

#%%

# Undertake an initial OE starting from NWP prior and see how close we get buoy truth using only the split window
Zr, Sr, Ar = optimal_estimates(Zan, K, SSan, SSen, Yn, Fn,usechan=[False,True,True])
xoe0 = np.array(Zr[0,:] ).squeeze()
sens0 = np.array(Ar[0,0] ).squeeze()*100
ux0 = np.sqrt(np.array(Sr[0,0] ).squeeze())

# Printout stats and plots
print('---------- STATS and PLOTS no bias correction using GBCS cal & initial covariances ------------')
stats0 = diagnostic_plots(xoe0, xt, solz, satz, lat, lon, time, elem, w, U, sens0)


#%%

# Adjust the simulations to match the starting point from skin-adjusted buoy SST
Y0, F0, Fx0, Fw0, Z0, K0 = make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    xt, t3, t4, t5, w, adj_for_x = True)

# Set up initial estimates of covariance matrices 
SSe0, SSa0 = initial_covs(sec, w, xatype = 'buoy', scale = 0.25) # using uncertainty appropriate to buoy as prior


# Set the coefficients we are going to optimise
#coef_list = [True, False, False, False, True, False, False, False, True, False, False, False]  # this is doing only offset coeffciients
coef_list = [True, True , False, False, True, True , False, False, True, True , False, False]  # this is doing only offset and emissivity 

# Set the number of strata on which to calculate a water vapour prior bias
divsg = 5

# List of arrays to pass all the counts and calibration counts
calinfo = [c3,cs3,cict3,lict3,c4,cs4,cict4,lict4,c5,cs5,cict5,lict5,nT]

# The initial estimate of prior TCWV bias and uncertainty, in strata
gamma0 = np.zeros(divsg)
ugamma0 = np.full(divsg, 0.2) # which is about 1% of the mean TCWV to start with


beta1, gamma1, gvals1, gc1, Sbeta1, Sgamma1 = update_beta_gamma3(F0, Fx0, Fw0, Z0, SSe0, SSa0, beta, coef_list, gamma0, \
                      w, divsg, 1000000, lut, calinfo, Sbeta*400, \
                      ugamma0,  accel = 5, extrapolate = True)
        # *X and accel are just to allow values to change more rapidly -- plots verify that it is still stable


l3r,t3r,tb3r,l4r,t4r,tb4r,l5r,t5r,tb5r,only2chan = calc_obs(calinfo, beta1)

#%%
# Some sanity checks
np.nanmean(t3r-t3)
np.sum(np.nanmean(tb3,axis=1)*(beta1-beta)[0:4])

np.nanmean(t4r-t4)
np.sum(np.nanmean(tb4,axis=1)*(beta1-beta)[4:8])

np.nanmean(t5r-t5)
np.sum(np.nanmean(tb5,axis=1)*(beta1-beta)[8:])


# Now do a new retrieval 
Y1, F1, Fx1, Fw1, Za1, K1 = make_matrices(f3+gc1*fw3/w, f4+gc1*fw4/w, f5+gc1*fw5/w, \
                                        fx3, fx4, fx5, fw3, fw4, fw5, x, \
    x+ np.mean(xt-x), t3r, t4r, t5r, w+gc1, adj_for_x = True, fix_ch3_nan = True)
nc = np.shape(Y1)[0]


# Set up initial estimates of covariance matrices for a first retrieval attempt
SSen, SSan = initial_covs(sec, w, xatype = 'nwp', scale = 0.25) # using uncertainty appropriate to prior

# Undertake an initial OE starting from NWP prior and see how close we get buoy truth
Zr, Sr, Ar = optimal_estimates(Za1, K1, SSan, SSen, Y1, F1)
xoe1 = np.array(Zr[0,:] ).squeeze()
sens1 = np.array(Ar[0,0] ).squeeze()*100
ux1 = np.sqrt(np.array(Sr[0,0] ).squeeze())

# Printout stats and plots
print('---------- STATS and PLOTS orrection using GBCS cal & initial covariances ------------')
stats1 = diagnostic_plots(xoe1, xt, solz, satz, lat, lon, time, elem, w, U, sens1)

#%%

# BT impact of change in coefficients

c3counts = [i for i in range(740, 991, 10)]
l3old = count2rad(c3counts,np.nanmean(cs3),np.nanmean(cict3),np.nanmean(lict3),0,3,beta[0:4])
t3old = rad2bt(l3old,3,lut)
l3new = count2rad(c3counts,np.nanmean(cs3),np.nanmean(cict3),np.nanmean(lict3),0,3,beta1[0:4])
t3new = rad2bt(l3new,3,lut)
plt.plot(c3counts,t3new-t3old)
plt.title('BT change as function of counts')
c4counts = [i for i in range(400, 860, 10)]
l4old = count2rad(c4counts,np.nanmean(cs4),np.nanmean(cict4),np.nanmean(lict4),0,4,beta[4:8])
t4old = rad2bt(l4old,4,lut)
l4new = count2rad(c4counts,np.nanmean(cs4),np.nanmean(cict4),np.nanmean(lict4),0,4,beta1[4:8])
t4new = rad2bt(l4new,4,lut)
plt.plot(c4counts,t4new-t4old)
c5counts = [i for i in range(380, 840, 10)]
l5old = count2rad(c5counts,np.nanmean(cs5),np.nanmean(cict5),np.nanmean(lict5),0,5,beta[8:])
t5old = rad2bt(l5old,5,lut)
l5new = count2rad(c5counts,np.nanmean(cs5),np.nanmean(cict5),np.nanmean(lict5),0,5,beta1[8:])
t5new = rad2bt(l5new,5,lut)
plt.plot(c5counts,t5new-t5old)
plt.legend(['Revised minus harmon., Ch3', 'Ch4', 'Ch5'])
plt.show()

plt.title('BT change as function of scene temperature')
plt.plot(t3new,t3new-t3old)
plt.plot(t4new,t4new-t4old)
plt.plot(t5new,t5new-t5old)
plt.legend(['Revised minus harmon., Ch3', 'Ch4', 'Ch5'])
plt.show()

plt.plot(t4r,gc1,'.')
plt.title('TCWV correction vs. BT')
plt.savefig('')

