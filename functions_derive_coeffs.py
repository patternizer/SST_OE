#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:46:33 2019

@author: chris
"""

import os
import datetime
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pylab as plt
import pickle
import dask

#%%

def decompose_S(Scov):
        
    U = np.asmatrix(np.diag(np.sqrt(np.diag(Scov))))
    R = U.I @ Scov @ U.I
    
    return U.astype('float32'), R.astype('float32')


#%%

def check_compatible_files(path, dirS, dirG):
    
    """
    Matches are coming in TWO files which are in TWO differently named 
    directories with identical filenames
    
    This function looks through the files in the directories passed to it
    and checks that all the names are the same
    and then checks that files of the same name have identical numbers
    of records
    
    path is the common root directory for the set of three
    dirS is the directory with source match information
    dirG is the directory with the added GBCS outputs

    outputs a list of compatible filenames if checks are passed  
    
    """
    
    pathG = path + dirG
    fG = np.sort([f for f in os.listdir(pathG) if f.endswith('.nc')])
       
    pathS = path + dirS
    fS = np.sort([f for f in os.listdir(pathS) if f.endswith('.nc')])

   
    n = np.max([fG.size, fS.size])
    
    matching = [False for i in range(n)]
    
    for f in range(n):
        matching[f] = (fS[f] == fG[f])
        
    if ~np.all(matching):
        use = np.where(np.logical_not(matching))
        print('Mismatching file names in directories')
        print([fS[use], fG[use]])
        return
    
    for i,f in enumerate(fS):
        dsG = xr.open_dataset(os.path.join(pathG,f))
        dsS = xr.open_dataset(os.path.join(pathS,f))
        matching[i] = (dsG.dims['record'] == dsS.dims['matchup_count'])
            
    if ~np.all(matching):
        use = np.where(np.logical_not(matching))
        print('Mismatching numbers of records in  these files')
        print([fG[use], fS[use]])
        return

    return fG[:]

#%%

    
def read_files(path, dirX, flist, dimstr = 'record', reduce = False, satcode = 'ma'):
    
    """
    Reads the netcdf files listed in flist from directory path+dirX
    keeps only the central 3x3 of the imagette
    """

    data = []
    
    for f in flist:
        
        ds = xr.open_dataset(os.path.join(path+dirX,f))
        if reduce:
            centre_dim = {d:[ds.dims[d]//2 -1,ds.dims[d]//2,ds.dims[d]//2 +1] \
                          for d in ds.dims if (d.endswith('nj') or d.endswith('ni') \
                                               or d.endswith(satcode+'_nx') or d.endswith(satcode+'_ny')) }
            ds = ds[centre_dim]
        #v=ds.load()
        
        try:
            ds1a = ds.drop('avhrr-'+satcode+'_dtime') # this is a work-around, as this variable makes
                        # the concat method fail for reasons I don't understand
        except:
            ds1a = ds

        try:
            ds1b = ds1a.drop_dims('insitu.ntime') # don't need these
        except:
            ds1b = ds1a
            
        data.append(ds1b)
        
    ds2 = xr.concat(data, dim=(dimstr))

    return ds2

#%%
   
def filter_matches(dsG, dsS, minpclr = 0.9,  sstminQL = 4, maxsza = 45., satcode = 'ma'):
    
    """
    returns a boolean
    keep the true records
    
    """
    nm = dsG.dims['record']
    cbox = np.size(dsG['gbcs.flags'][0,:,0])//2    
    keep = [True for i in range(nm)]
    
    pclr = np.array(dsG['gbcs.p_clear.max'].min(dim=('ni','nj')))
    pclr[np.isnan(pclr)] = 0
    keep = np.logical_and(keep, pclr > minpclr)

    QCin = np.array(dsS['drifter-sst_insitu.qc1']).astype('int').squeeze()
    keep = np.logical_and(keep, QCin == 0)
 
    QL = np.array(dsG['gbcs.quality']).astype('int')[:,cbox,cbox]
    keep = np.logical_and(keep, QL >= sstminQL)
    
    szain = np.array(dsS['avhrr-'+satcode+'_satellite_zenith_angle'][:,cbox,cbox])
    keep = np.logical_and(keep, szain < maxsza)

    flagS = np.array(dsG['gbcs.flags'][:,cbox,cbox]).astype('int')
    keep = np.logical_and(keep, np.logical_or(flagS == 128, flagS == 256))

    Kcheck = np.array(dsG['ffm.dbt_dtcwv_4'])[:,0,0]
    keep = np.logical_and(keep, np.isfinite(Kcheck))

    Kcheck = np.array(dsG['ffm.dbt_dtcwv_5'])[:,0,0]
    keep = np.logical_and(keep, np.isfinite(Kcheck))

    Kcheck = np.array(dsG['ffm.dbt_dsst_4'])[:,0,0]
    keep = np.logical_and(keep, np.isfinite(Kcheck))

    Kcheck = np.array(dsG['ffm.dbt_dsst_5'])[:,0,0]
    keep = np.logical_and(keep, np.isfinite(Kcheck))

    return keep

#%%

    
def extract_vars(dsG, dsS, satcode = 'ma'):
    
    data = []
    cbox = np.size(dsG['gbcs.flags'][0,:,0])//2
    
    mvars = ['gbcs.flags', 'gbcs.p_clear.max']
    for v in mvars: data.append(np.array(dsG[v][:,cbox,cbox]))
    
    mvars = [ 'ffm.brightness_temperature_3b',\
             'ffm.brightness_temperature_4', 'ffm.brightness_temperature_5', \
             'ffm.dbt_dsst_3b','ffm.dbt_dsst_4','ffm.dbt_dsst_5',\
             'ffm.dbt_dtcwv_3b','ffm.dbt_dtcwv_4', 'ffm.dbt_dtcwv_5',\
             'nwp.sst']
    for v in mvars: data.append(np.array(dsG[v][:,0,0]))    
    
    bvars = ['avhrr-'+satcode+'_ch1', 'avhrr-'+satcode+'_ch2', \
             'avhrr-'+satcode+'_ch3a', 'avhrr-'+satcode+'_ch3b', \
             'avhrr-'+satcode+'_ch4', 'avhrr-'+satcode+'_ch5',\
             'avhrr-'+satcode+'_ch1_earth_counts', 'avhrr-'+satcode+'_ch2_earth_counts', \
             'avhrr-'+satcode+'_ch3a_earth_counts', 'avhrr-'+satcode+'_ch3b_earth_counts', \
             'avhrr-'+satcode+'_ch4_earth_counts', 'avhrr-'+satcode+'_ch5_earth_counts',\
             'avhrr-'+satcode+'_ch3b_bbody_counts', \
             'avhrr-'+satcode+'_ch4_bbody_counts', 'avhrr-'+satcode+'_ch5_bbody_counts',\
             'avhrr-'+satcode+'_ch3b_space_counts', \
             'avhrr-'+satcode+'_ch4_space_counts', 'avhrr-'+satcode+'_ch5_space_counts',\
             'avhrr-'+satcode+'_ict_temp', 'avhrr-'+satcode+'_orbital_temperature',\
             'avhrr-'+satcode+'_orbital_temperature_nlines',\
             'avhrr-'+satcode+'_prt_1', 'avhrr-'+satcode+'_prt_2', 'avhrr-'+satcode+'_prt_3', 'avhrr-'+satcode+'_prt_4', \
             'avhrr-'+satcode+'_solar_zenith_angle', 'avhrr-'+satcode+'_satellite_zenith_angle',\
             'avhrr-'+satcode+'_lat', 'avhrr-'+satcode+'_lon']

    for v in bvars: data.append(np.array(dsS[v][:,cbox,cbox])) 
        
    dum = np.array(dsS['avhrr-'+satcode+'_acquisition_time'][0:-1,cbox,cbox])
    dum = np.append(dum, np.array(dsS['avhrr-'+satcode+'_acquisition_time'][-1,cbox,cbox]) )
    # This is a cludge because of dask madness
    data.append(dum)
    
    
    bvars = ['avhrr-'+satcode+'_x', 'avhrr-'+satcode+'_y']
    for v in bvars: data.append(np.array(dsS[v][:]))     
    
    bvars = [ 'avhrr-'+satcode+'_nwp_total_column_water_vapour']   
    for v in bvars: data.append(np.array(dsS[v][:,0,0])) 

    wind = np.sqrt(np.array( ( (dsS['avhrr-'+satcode+'_nwp_10m_east_wind_component'][:,0,0])**2 + \
             (dsS['avhrr-'+satcode+'_nwp_10m_north_wind_component'][:,0,0])**2  )))
    
    sec = 1./np.cos(np.deg2rad(np.array(dsS['avhrr-'+satcode+'_satellite_zenith_angle'][:,cbox,cbox])))

    data.append(wind[:])
    data.append(sec[:])
    
    pvars = ['drifter-sst_insitu.sea_surface_temperature']
    for v in pvars: data.append(np.array(dsS[v][:]).squeeze())      

    return data

#%%
    
def run_checking_plots(mpclr, f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    y1, y2, y3a, y3, y4, y5, solz, satz, lat, lon, time, elem, line, w, runtag):
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 2, 1)    
    plt.hist(mpclr*100)
    plt.title('Metop P Clear %')
    ax2 = fig.add_subplot(2, 2, 2)    
    plt.plot(elem,satz,'.')
    plt.title('Elem vs SZA') 
    ax3 = fig.add_subplot(2, 2, 3)    
    plt.scatter(w,y4-y5,c=satz,s=1.5)
    plt.title('Y4 - Y5 vs TCWV by SZA')    
    ax4 = fig.add_subplot(2, 2, 4)    
    plt.hist(w)
    plt.title('TCWV')
    plt.savefig(runtag+'tcwv.png')
    plt.close('all')
    
    # Locations
    fig = plt.figure(figsize=(12,8))
    axs = fig.add_subplot(2, 2, 3)
    h2 = axs.hist2d(lon, lat, bins=90, range=[[-180,180],[-90,90]], vmin=0)
    plt.title('histogram of (lon,lat)')
    ax1 = fig.add_subplot(2, 2, 4)
    hy = ax1.hist(lat, 90, [-90,90], orientation='horizontal')
    plt.title('histogram of lat')
    ax1.set_yticklabels([])
    axs.get_shared_y_axes().join(axs,ax1)
    ax2 = fig.add_subplot(2, 2, 1)
    hx = ax2.hist(lon, 90, [-180,180])
    plt.title('histogram of lon')
    ax2.set_xticklabels([])
    axs.get_shared_x_axes().join(axs,ax2)
    axs.set_xlim(-180,180)
    axs.set_ylim(-90,90)
    ax3 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.set_global()
    ax3.plot(lon, lat, ',')
    plt.title('locations')
    fig.tight_layout()
    plt.savefig(runtag+'locations.png')
    plt.close('all')

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 2, 1)    
    plt.scatter(x,y3-f3,c=w,s=1.5)
    plt.title('Y3-F3 vs SST by W')
    ax2 = fig.add_subplot(2, 2, 2)    
    plt.scatter(w,y3-f3,c=satz,s=1.5)
    plt.title('Y3-F3 vs TCWV by SATZ') 
    ax3 = fig.add_subplot(2, 2, 3)    
    plt.plot(lat,y3-f3, '.')
    plt.title('Y3-F3 vs lat')     
    ax4 = fig.add_subplot(2, 2, 4)    
    plt.plot(elem, y3-f3,'.')
    plt.title('Y3-F3 vs elem')
    plt.savefig(runtag+'y3-f3-vs-elem.png') 
 
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 2, 1)    
    plt.scatter(x,y4-f4,c=w,s=1.5)
    plt.title('Y4-F4 vs SST by W')
    ax2 = fig.add_subplot(2, 2, 2)    
    plt.scatter(w,y4-f4,c=satz,s=1.5)
    plt.title('Y4-F4 vs TCWV by SATZ') 
    ax3 = fig.add_subplot(2, 2, 3)    
    plt.plot(lat,y4-f4, '.')
    plt.title('Y4-F4 vs lat')     
    ax4 = fig.add_subplot(2, 2, 4)    
    plt.plot(elem, y4-f4,'.')
    plt.title('Y4-F4 vs elem')
    plt.savefig(runtag+'y4-f4-vs-elem.png') 
    plt.close('all')

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 2, 1)    
    plt.scatter(x,y5-f5,c=w,s=1.5)
    plt.title('Y5-F5 vs SST by W')
    ax2 = fig.add_subplot(2, 2, 2)    
    plt.scatter(w,y5-f5,c=satz,s=1.5)
    plt.title('Y5-F5 vs TCWV by SATZ') 
    ax3 = fig.add_subplot(2, 2, 3)    
    plt.plot(lat,y5-f5, '.')
    plt.title('Y5-F5 vs lat')     
    ax4 = fig.add_subplot(2, 2, 4)    
    plt.plot(elem, y5-f5,'.')
    plt.title('Y5-F5 vs elem')
    plt.savefig(runtag+'y5-f5-vs-elem.png') 
    plt.close('all')

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2, 2, 1)    
    plt.plot(y3-y5,y4-y5,'.')
    plt.title('Y4-Y5 v Y3-Y5')
    ax2 = fig.add_subplot(2, 2, 2)    
    plt.plot(time,satz,'.')
    plt.title('time v SATZ') 
    ax3 = fig.add_subplot(2, 2, 3)    
    plt.plot(time,lat, '.')
    plt.title('lat vs time')     
    ax4 = fig.add_subplot(2, 2, 4)    
    plt.scatter(w, fw5/w,c=x,s=1.5)
    plt.title('dF5dw vs TCWV by SST')
    plt.savefig(runtag+'df5dw-vs-tcwv-by-sst.png') 
    plt.close('all')

    return
    
#%%

def read_harm_init(Hpath, current_sensor):
    
    # if current_sensor == 'ma': current_sensor = 'm02'
    # The Harmonisation file for 3.7 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_37.nc')
    names = [str(hp['sensor_name'][i])[44:47] for i in range(2,len(hp['sensor_name']))]
    if current_sensor == 'ma':
        isen = names.index('m02')
    else:
        isen = names.index(current_sensor) # which sensor in the series to get the coefficients for
    
    co3 = np.array(hp['parameter'][isen*3:(isen*3+3)])
    co3 = np.array([co3[0],co3[1],0.0,co3[2]])  # add zero non-linearity coefficient in right place
    uco3 = np.array(hp['parameter_uncertainty'][isen*3:(isen*3+3)])
    uco3 = np.array([uco3[0],uco3[1],1e-12,uco3[2]])  # put a tiny uncertainty on the zero value for non-linearity
    Sco3 = np.array(hp['parameter_covariance_matrix'])[isen*3:(isen*3+3),isen*3:(isen*3+3)]
    Sco3 = np.insert(Sco3, 2, 0, axis = 0)
    Sco3 = np.insert(Sco3, 2, 0, axis = 1)
    Sco3[2,2] = uco3[2]**2
    
    # for 11 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_11.nc')
    co4 = np.array(hp['parameter'][isen*4:(isen*4+4)])
    uco4 = np.array(hp['parameter_uncertainty'][isen*4:(isen*4+4)])
    Sco4 = np.array(hp['parameter_covariance_matrix'])[isen*4:(isen*4+4),isen*4:(isen*4+4)]
    
    # for 12 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_12.nc')
    co5 = np.array(hp['parameter'][isen*4:(isen*4+4)])
    uco5 = np.array(hp['parameter_uncertainty'][isen*4:(isen*4+4)])
    Sco5 = np.array(hp['parameter_covariance_matrix'])[isen*4:(isen*4+4),isen*4:(isen*4+4)]

    beta = np.concatenate((co3,co4,co5))
    ubeta = np.concatenate((uco3,uco4,uco5))
    Sbeta = np.diag(ubeta)
    Sbeta[0:4,0:4] = Sco3
    Sbeta[4:8,4:8] = Sco4
    Sbeta[8:,8:] = Sco5    
    pnorm = np.array([hp['sensor_equation_config'][isen*2:(isen*2+2) ]]).squeeze()

    return beta, ubeta, Sbeta, pnorm

#%%

def read_harm_init2(Hpath, current_sensor):
    
    # The Harmonisation file for 3.7 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_37.nc')
    names = [str(hp['sensor_name'][i])[44:47] for i in range(2,len(hp['sensor_name']))]
    if current_sensor == 'ma':
        isen = names.index('m02')
    else:
        isen = names.index(current_sensor) # which sensor in the series to get the coefficients for
    
    co3 = np.array(hp['parameter'][isen*3:(isen*3+3)])
    uco3 = np.array(hp['parameter_uncertainty'][isen*3:(isen*3+3)])
    Sco3 = np.array(hp['parameter_covariance_matrix'])[isen*3:(isen*3+3),isen*3:(isen*3+3)]
    
    # for 11 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_11.nc')
    co4 = np.array(hp['parameter'][isen*4:(isen*4+4)])
    uco4 = np.array(hp['parameter_uncertainty'][isen*4:(isen*4+4)])
    Sco4 = np.array(hp['parameter_covariance_matrix'])[isen*4:(isen*4+4),isen*4:(isen*4+4)]
    
    # for 12 um
    hp = xr.open_dataset(Hpath+'FIDUCEO_Harmonisation_Data_12.nc')
    co5 = np.array(hp['parameter'][isen*4:(isen*4+4)])
    uco5 = np.array(hp['parameter_uncertainty'][isen*4:(isen*4+4)])
    Sco5 = np.array(hp['parameter_covariance_matrix'])[isen*4:(isen*4+4),isen*4:(isen*4+4)]

    beta = np.concatenate((co3,co4,co5))
    ubeta = np.concatenate((uco3,uco4,uco5))
    Sbeta = np.diag(ubeta)
    Sbeta[0:3,0:3] = Sco3
    Sbeta[3:7,3:7] = Sco4
    Sbeta[7:,7:] = Sco5

    return beta, ubeta, Sbeta


#%%

def count2rad(Ce,Cs,Cict,Lict,Tinst,channel,coef):
    '''
    NB: Tinst is the normalized temperature (T - T_mean)/T_std
    NB: Additional WV term added in the measurement equations
    '''
    a1,a2,a3,a4 = coef        
    
    L = a1 + ((Lict * (0.985140 + a2)) / (Cict - Cs) + a3 * (Ce - Cict)) * (Ce - Cs) + a4 * Tinst 

    return L

def count2rad2(Ce,Cs,Cict,Lict,Tinst,channel,coef):
    '''
    NB: Tinst is the normalized temperature (T - T_mean)/T_std
    NB: Additional WV term added in the measurement equations
    '''
    if channel == 3 : 
        a1,a2,a4 = coef
        a3=0
    else: 
        a1,a2,a3,a4 = coef        
    
    L = a1 + ((Lict * (0.985140 + a2)) / (Cict - Cs) + a3 * (Ce - Cict)) * (Ce - Cs) + a4 * Tinst 

    return L


def read_in_LUT(avhrr_sat, lutdir = './'):
    LUT = {}
    all_lut_radiance_dict = np.load(lutdir+'lut_radiance.npy', encoding='bytes', allow_pickle=True).item()
    all_lut_BT_dict = np.load(lutdir+'lut_BT.npy', encoding='bytes', allow_pickle=True).item()
    try:
        LUT['L'] = all_lut_radiance_dict[avhrr_sat][:]
        LUT['BT'] = all_lut_BT_dict[avhrr_sat][:]
    except:
        print("Sensor for AVHRR does not exist: ", avhrr_sat)
        print('Choose from:', all_lut_radiance_dict.keys())

    return LUT

def rad2bt(L,channel,lut):
    BT = np.interp(L,lut['L'][:,channel],lut['BT'][:,channel],left=np.nan,right=np.nan)
    return BT

def bt2rad(BT,channel,lut):
    L = np.interp(BT,lut['BT'][:,channel],lut['L'][:,channel],left=np.nan,right=np.nan)
    return L

def drad_da2(Ce,Cs,Cict,Lict,Tinst,channel):

    drad_da1 = Lict/Lict
    drad_da2 = Lict*(Ce-Cs)/(Cict - Cs)
    drad_da3 = (Ce - Cict) * (Ce - Cs)
    drad_da4 = Tinst
    if channel == 3:
        return np.array([drad_da1,drad_da2,drad_da4])
    else:
        return np.array([drad_da1,drad_da2,drad_da3,drad_da4])

def drad_da(Ce,Cs,Cict,Lict,Tinst,channel):

    drad_da1 = Lict/Lict
    drad_da2 = Lict*(Ce-Cs)/(Cict - Cs)
    drad_da3 = (Ce - Cict) * (Ce - Cs)
    drad_da4 = Tinst
    return np.array([drad_da1,drad_da2,drad_da3,drad_da4])
               
def dbtdL(T,channel,lut):
    from scipy import interpolate
    grads = np.array(np.gradient(lut['L'][:,channel],lut['BT'][:,channel])[:])
    f = interpolate.interp1d(lut['BT'][:,channel],grads)
    return 1/f(T)

def make_matrices(f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    xret, y3, y4, y5, w, solz, adj_for_x = True, fix_ch3_nan = True, exc_ch3_day = True, drop_day = False):

    cf3, cf4, cf5, cfx3, cfx4, cfx5, cfw3, cfw4, cfw5, cx, \
    cxret, cy3, cy4, cy5, cw, csolz = np.copy([f3, f4, f5, fx3, fx4, fx5, fw3, fw4, fw5, x, \
    xret, y3, y4, y5, w, solz]) 

    if drop_day:
        keep = (solz > 90.)
        cf3, cf4, cf5, cfx3, cfx4, cfx5, cfw3, cfw4, cfw5, cx, \
                  cxret, cy3, cy4, cy5, cw, csolz = [f[keep] for f in 
                  [cf3, cf4, cf5, cfx3, cfx4, cfx5, cfw3, cfw4, cfw5, cx, \
                  cxret, cy3, cy4, cy5, cw, csolz]]

    if fix_ch3_nan:
        isnan = np.isnan(cy3)
        cfx3[isnan] = 0.0 # giving no sensitivity to this channel
        cfw3[isnan] = 0.0

    if exc_ch3_day:
        isexc = (csolz < 90.)
        cy3[isexc] = np.nan # purely nominal typical value, not to be used in retrieval
        cfx3[isexc] = 0.0 # giving no sensitivity to this channel
        cfw3[isexc] = 0.0
            
    Y  = np.asmatrix((cy3, cy4, cy5))
    F  = np.asmatrix((cf3, cf4, cf5))
    Fx = np.asmatrix((cfx3, cfx4, cfx5))
    Fw = np.asmatrix((cfw3/cw, cfw4/cw, cfw5/cw)) # converting dY/(dw/w) into dYdw
    Za = np.asmatrix((cx, cw))      
    
    if adj_for_x: 
        for i in range(np.size(cx)):
            F[:,i] += (cxret - cx)[i]*Fx[:,i]
        Za[0,:] = cxret
    
    K = []
    for i in range(np.shape(Y)[1]):
        K.append(np.concatenate((Fx[:,i], Fw[:,i]), axis = 1))

    return Y, F, Fx, Fw, Za, K # K is a list of matrices

def initial_covs(sec, w, xatype = 'd3', xatypes = {'buoy':0.2, 'd3':0.15, 'd2':0.25, 'nwp':0.55,  'clim':1.0} , scale = 1.):
    
    nm = np.size(sec)
    
    ux = xatypes[xatype]

    Se0 = np.zeros((3,3,nm))
    Sa0 = np.zeros((2,2,nm))
    
    for i in range(0,nm):
        Se0[:,:,i] = np.diag([0.10**2 + (0.15*sec[i])**2, 0.10**2 + (0.15*sec[i])**2, \
                                   0.10**2 + (0.15*sec[i])**2])*scale
        Sa0[:,:,i] =  np.diag([ux**2, ((1.6+0.04*w[i])**2)*scale])
        
    return Se0, Sa0
       
def optimal_estimate(ZZa, KK, SSa, SSe, YY, FF):
    
    SSeI = SSe.I
    
    SS = (KK.T @ SSeI @ KK + SSa.I).I 
    
    GG = SS @ KK.T @ SSeI
    
    ZZ = ZZa + GG @ (YY - FF)
    
    AA = GG @ KK
    
    return ZZ, SS, AA

def optimal_estimates(Z, K, Sa, Se, Y, F, usechan = -1):
    
    nm = np.size(Z[0,:])
    Zr = np.asmatrix(np.zeros((np.shape(Z))))
    Sr  = np.zeros((np.shape(Sa)))
    Ar  = np.zeros((np.shape(Sa)))
    
    if usechan == -1 : usechan = [True, True, True]
    
    for i in range(nm):   
        
        cy = np.array(np.copy(Y[:,i])).squeeze()
        lusechan = np.logical_and(usechan, np.isfinite(cy))        
        YY = Y[lusechan,i]    

        SSe = np.asmatrix(Se[:,:,i])[lusechan,:][:,lusechan]
        SSa = np.asmatrix(Sa[:,:,i])
            
        ZZa = Z[:,i]    
        FF = F[lusechan,i]    
        KK = K[i][lusechan,:] # K passed in must be a list of matrices
                
        #SSeI = SSe.I
        
        SS = (KK @ SSa @ KK.T + SSe).I 
        
        GG = SSa @ KK.T @ SS
        
        dY = (YY - FF)
        
        ZZ = ZZa + GG @ dY
        
        AA = GG @ KK
        
        SSo = SSa - AA @ SSa
        
        #SSaI = (KK @ SSa @ KK.T + SSe).I
        
        #SSdy = SSe @ SSaI @ SSe
        
        #dYr = KK @ (ZZ - ZZa)
        
        #chi2r = dYr.T @ SSdy.I @ dYr # chi2 for whether the retrieval fits the measurement
        
        Zr[:,i] = ZZ
        Sr[:,:,i] = SSo
        Ar[:,:,i] = AA
        
    return Zr, Sr, Ar
    
def summary_stats(data, stratified = False, sdata = -1, stitle = 'unspecified',
                  strata = -1, title = 'data', units = 'K', 
                  precision = 3, exact = False, output = False):
    
    from statsmodels import robust as rb
    p = precision
    
    ns = np.max((np.size(strata)-1+exact,1))
 
    outdata = np.zeros((7,ns))
    
    if stratified == False:
        print('Summary statistics of '+title) 
        print('N        Mean     SD       Median   RSD / '+units)
        outdata[:,0] = [data.size, np.mean(data), np.std(data), \
                           np.median(data),  rb.mad(data), np.mean(sdata), \
                           np.std(data)/np.sqrt((np.max([data.size-1.5,1])))]
        print("{: <8d}".format(data.size), \
        "{: <8.{}f}".format(np.mean(data), p), \
        "{: <8.{}f}".format(np.std(data),p),\
        "{: <8.{}f}".format(np.median(data),p),\
        "{: <8.{}f}".format(rb.mad(data),p) )
    else:
        print( 'Summary statistics of '+title  + ', stratified by: '+stitle )
        print( 'N        Mean     SD       Median   RSD / '+units )
        if ns == 1:
            print( 'NB not stratified')
            outdata[:,0] = [data.size, np.mean(data), np.std(data), \
                           np.median(data),  rb.mad(data), np.mean(sdata), \
                           np.std(data)/np.sqrt((np.max([data.size-1.5,1]))) ]
            print( "{: <8d}".format(data.size), \
            "{: <8.{}f}".format(np.mean(data), p), \
            "{: <8.{}f}".format(np.std(data),p),\
            "{: <8.{}f}".format(np.median(data),p),\
            "{: <8.{}f}".format(rb.mad(data),p) )
        else:
            for n in range(0,ns):
                if exact == False:
                    instratum = ((sdata >= strata[n]) & (sdata < strata[n+1]))
                    print( 'Stratum: ', "{: <8.{}f}".format(strata[n], p), ' to ', "{: <8.{}f}".format(strata[n+1], p))
                    outdata[:,n] = [data[instratum].size, np.mean(data[instratum]), np.std(data[instratum]), \
                           np.median(data[instratum]),  rb.mad(data[instratum]), np.mean(sdata[instratum]), \
                           np.std(data[instratum])/np.sqrt((np.max([data[instratum].size-1.5,1]))) ]
                else:
                    instratum = (sdata == strata[n]) 
                    print( 'Stratum: ', strata[n])
                    outdata[:,n] = [data[instratum].size, np.mean(data[instratum]), np.std(data[instratum]), \
                           np.median(data[instratum]),  rb.mad(data[instratum]), strata[n], \
                           np.std(data[instratum])/np.sqrt((np.max([data[instratum].size-1.5,1]))) ]
                print( "{: <8d}".format(data[instratum].size), \
                "{: <8.{}f}".format(np.mean(data[instratum]), p), \
                "{: <8.{}f}".format(np.std(data[instratum]),p),\
                "{: <8.{}f}".format(np.median(data[instratum]),p),\
                "{: <8.{}f}".format(rb.mad(data[instratum]),p)      )
    
    if output:
        return outdata
    else:
        return

def run_summary_stats(var, title, stratvar, divs, stitle):
    
    strata = np.percentile(stratvar, [np.round(i*1000./divs)/10. for i in range(divs+1)] ) 

    out = summary_stats(var, stratified = True, sdata = stratvar, stitle = stitle,
                  strata = strata, title = title, units = 'K', 
                  precision = 3, exact = False, output = True)
    
    return out

def diagnostic_plots(runtag, xret, xd3, solz, satz, lat, lon, time, elem, \
                     w, U, sens, title = 'AVHRR - Buoy', use_mean = False):
    
    if use_mean == True:
        ind = 1
    else:
        ind = 3
    
    
    # assumes time is float years
    
    invars = []
    invars.append([xret-xd3, title, satz, 10, 'Sat Zen',w,'TCWV'])
    invars.append([xret-xd3, title, lat, 10, 'Latitude',solz,'Sol ZA'])
    invars.append([xret-xd3, title, elem, 10, 'Element',w,'TCWV'])
    invars.append([xret-xd3, title, w, 10, 'TCWV',satz,'Sat ZA'])
    invars.append([xret-xd3, title, U, 10, 'Wind',lat,'Lat'])
    invars.append([xret-xd3, title, xd3, 10, 'SST',w,'TCWV'])
    invars.append([xret-xd3, title, solz, 10, 'Sol ZA',sens,'Sens.'])
    invars.append([xret-xd3, title,  time.astype('int64')/1e9/3600/24/365.25+1970, 10, 'Time',lat,'Latit'])

    outvars = []
    k = 0
    for i in invars: 
        k += 1
        j = run_summary_stats(i[0],i[1],i[2],i[3],i[4])
        outvars.append(j)
        fig,ax = plt.subplots()
        plt.scatter(i[2],i[0],c=i[5],s=1.5)
        plt.ylim(-1.0,1.0)
        plt.title(i[1]+' vs '+i[4]+' by '+i[6])
        plt.plot(j[5,:],j[ind,:],color='red')
        plt.plot(j[5,:],j[ind,:]+j[ind+1,:],'-.',color='red')
        plt.plot(j[5,:],j[ind,:]-j[ind+1,:],'-.',color='red')
        pltstr = runtag+str(k)+'.png'
        plt.savefig(pltstr)
        plt.close('all')
 
    summary_stats(xret-xd3, title = title)

    fig,ax = plt.subplots()
    plt.hist((xret-xd3)*100)
    plt.title(title+', cK')  
    plt.savefig(runtag+'cK.png')
    plt.close('all')
    
    summary_stats(sens, title = 'Sensitivity')
    
    return outvars
    
def calc_obs(calinfo, tict, beta, lut):

    c3,cs3,cict3,lict3,c4,cs4,cict4,lict4,c5,cs5,cict5,lict5,nT = calinfo

    # first, turn Tinst into normalised Tinst
    # nT = (tinst - 286.125823)/0.049088 
    # placeholder for now -- **** these numbers for MetopA need to be generalised to other sensors  ****
    
    # created lict values
    lict3, lict4, lict5 = bt2rad(tict,3,lut), bt2rad(tict,4,lut), bt2rad(tict,5,lut)
    
    # calculate the new "observed" BTs
    l3 = count2rad(c3,cs3,cict3,lict3,nT,3,beta[0:4])
    t3 = rad2bt(l3,3,lut)
    tb3 = dbtdL(t3,3,lut) * drad_da(c3,cs3,cict3,lict3,nT,3)
    
    only2chan = np.where(t3 == np.nan) # flag for when only 11 and 12 are present
    

    l4 = count2rad2(c4,cs4,cict4,lict4,nT,4,beta[4:8])
    t4 = rad2bt(l4,4,lut)
    tb4 = dbtdL(t4,4,lut) * drad_da(c4,cs4,cict4,lict4,nT,4)
    
    l5 = count2rad2(c5,cs5,cict5,lict5,nT,5,beta[8:])
    t5 = rad2bt(l5,5,lut)
    tb5 = dbtdL(t5,5,lut) * drad_da(c5,cs5,cict5,lict5,nT,5)
    
    return l3,t3,tb3,l4,t4,tb4,l5,t5,tb5, only2chan
   
def update_beta_gamma3(runtag, F, Fx, Fw, Z, Se, Sa, betai, coef_list, gammai, \
                      auxg, divsg, ni, lut, calinfo, betaSi, \
                      ugammai,  accel = 1, \
                      verbose = True, makeplot = True, tag = '', \
                      extrapolate = False):
    
    """
    * the beta (calibration coefficient) estimation has no subdivisions since coefficients
    are for all observing situations
    
    * the gamma (water vapour prior correction) is done on divsg strata of the variable auxg  
    
    
    * This version: make the coefficients to be optimised selectable and deal carefully with 
    2 vs 3 channel cases
    
    coef_list is of length len(betai) and of form [True, False, True, ...]
    
    It is assumed that the same measurement equation applies to all channels, and use
    coef_list to suppress any terms that are not used for a given channel
    
    """
    
    c3,cs3,cict3,lict3,c4,cs4,cict4,lict4,c5,cs5,cict5,lict5,nT = calinfo
    
    nm = np.size(F[0,:]) # all the others are assumed to be consistent in size
    nc = np.size(F[:,0]) # number of channels
    
    nz = np.size(Z[:,0]) # number of geophysical state variables
        
    nb = np.sum(coef_list)  # number of coefficients that will be retrieved
    
    
    ng = 1 # the number of state variables with a bias correction -- just water vapour presently
    
    divsg = np.int(divsg) # in case not passed as intege
    if divsg > 1: 
        stratg = True
    else:
        stratg = False
    
    
    if ni < (divsg+1)*3000: 
        ni = (divsg+1)*3000
        if verbose: 
            print('Number of iterations increased to '+ str(ni))
    
    # the matrix to hold the results for the set of randomly selected matches
    iZr  = np.asmatrix(np.zeros((nz+nb+ng,ni)))


    if stratg:  # set limits of each stratum
        lauxg = np.percentile(auxg, [np.round(i*1000./divsg)/10. for i in range(divsg+1)] ) 
        lauxg[0] *= 0.999
        lauxg[-1] *= 1.001
    else:
        lauxg = np.percentile(auxg, [0.0, 100.0])

    # the matrix to hold the bin index for each retrieval, gamma auxvars
    bini = np.zeros(ni).astype('int') 
    
    # to hold per bin the updating error covariance estimate of the beta and gamma values
    betaS = betaSi[:,coef_list][coef_list,:] # initialise with input value
 
    gammaS = np.zeros((ng,ng,divsg)) #   divs
    for i in range(divsg): gammaS[:,:,i]  =  ugammai[i]**2
    
    #to hold the evolving beta values
    betaBC = np.copy(betai[coef_list])
    betac = betai.copy() 
        
    #to hold the evolving gamma values
    if gammai.shape[0] != (divsg):
        if verbose: print('Warning: Not using the passed in gamma matrix')
        gammaBC = np.zeros((divsg))
    else:
        gammaBC = np.copy(gammai)

    
    for i in range(0, ni):
        
        if verbose:
            if (i*5) % ni < 5: print((100*i)//ni, '% done')
    
        # First choose a random index from the MDN
        j = np.int(np.random.uniform(0,nm-0.5))

        if stratg:
            binjg = np.max(np.where(lauxg < auxg[j]))
        else:
            binjg = 0
        bini[i] = binjg
        
        zbc = gammaBC[binjg] # the current prior bias estimate
            
        # Specify the extended prior and OE matrices
        ZZa = np.concatenate(([Z[0,j]], [Z[1,j]+zbc], betaBC, [zbc]))
        ZZa = np.matrix(ZZa).T # need to generalize if using more state variables
    
        # Make the obs and their derivatives from the Counts  

        betac[coef_list] = betaBC
        t3 = rad2bt(count2rad(c3[j],cs3[j],cict3[j],lict3[j],nT[j],3,betac[0:4]),3,lut)
        tb3 = dbtdL(t3,3,lut) * drad_da(c3[j],cs3[j],cict3[j],lict3[j],nT[j],3)
        t4 = rad2bt(count2rad(c4[j],cs4[j],cict4[j],lict4[j],nT[j],4,betac[4:8]),4,lut)
        tb4 = dbtdL(t4,4,lut) * drad_da(c4[j],cs4[j],cict4[j],lict4[j],nT[j],4)
        t5 = rad2bt(count2rad(c5[j],cs5[j],cict5[j],lict5[j],nT[j],5,betac[8:]),5,lut)  
        tb5 = dbtdL(t5,5,lut) * drad_da(c5[j],cs5[j],cict5[j],lict5[j],nT[j],5)
        
        usechan = [~np.isnan(t3), True, True]
        
        
        YY = np.asmatrix([t3,t4,t5])[0,usechan].T 
        
        FF = F[usechan,j] + zbc*Fw[usechan,j]
        
        k4 = np.zeros(4)
        kb = np.vstack((np.concatenate((-tb3,k4,k4)), \
                             np.concatenate((k4,-tb4,k4)), \
                             np.concatenate((k4,k4,-tb5))))
        kb = kb[usechan,:]
        kb = kb[:,coef_list]

        KK = np.concatenate((Fx[usechan,j], Fw[usechan,j], kb, Fw[usechan,j]), axis = 1)
 
        SSe = np.asmatrix(Se[:,:,j].squeeze())[:,usechan][usechan,:]
       
        SSa = np.asmatrix(np.zeros((nz+nb+ng,nz+nb+ng)))
        SSa[:nz,:nz] = np.asmatrix(Sa[:,:,j].squeeze())
        SSa[nz:-ng,nz:-ng] = np.asmatrix(betaS[ :, :]) # the bias extimate error covariance from previous run
        SSa[-ng:,-ng:] = np.asmatrix(gammaS[ :, :, binjg]) # the bias extimate error covariance from previous run
        #SSa[nz:,nz:] = np.asmatrix(paramS)
        
        iZr[:, i], S, A = optimal_estimate(ZZa, KK, SSa, SSe, YY, FF)
        

        betaBC[:] = betaBC[:] + accel* (np.asarray(iZr[nz:-ng,i]).flatten() - betaBC[:] )
        
        gammaBC[binjg] = gammaBC[binjg] + accel*(np.asarray(iZr[-ng:,i]).flatten()-gammaBC[binjg])

        betaS[:, :] = S[nz:-ng,nz:-ng]
        gammaS[:, :, binjg] = S[-ng:,-ng:]
        #paramS = S[nz:,nz:]
  
    betac[coef_list] = np.array(np.mean(iZr[nz:-ng,:][:,-3000:],axis=1)).squeeze() 
                             
    gammaout = np.zeros((divsg))
    for i in range(divsg): gammaout[i] = np.mean(iZr[-ng:,bini[:]==i][:,-3000:],axis=1)
                        
    gvals = np.array([np.mean(auxg[np.logical_and(auxg>=lauxg[i], auxg< lauxg[i+1])]) \
                              for i in range(divsg) ]).flatten()   

    
    if verbose:

        print('Beta parameters')
        print(betac)  
        print('Mid-bin-values of stratifying variable for gamma')
        print(gvals)
        print('Gamma parameters per stratum')
        print(gammaout)  

        
    if makeplot:
        for i in range(nb):
            fig,ax = plt.subplots()
            plt.plot(np.asarray(iZr[nz+i,:]).flatten())
            plt.title('Coefficient'+str(i))            
            pltstr = runtag+'Coefficient'+'_'+str(i)+'.png'  
            plt.savefig(pltstr)
            plt.close('all')

        for i in range(divsg): 
            fig,ax = plt.subplots()
            plt.plot(np.asarray(iZr[-ng,bini[:]==i]).flatten())
            # plt.ylim((-0.5,0.5))
            plt.title('Gamma Correction Convergence')
            # plt.legend((str(gvals.astype('int'))))
            # pltstr = 'plot_from_update_gamma'+tag+'_stratum_'+str(i)+'.png'
            pltstr = runtag+'plot_from_update_gamma_stratum_'+str(i)+'.png'
            plt.savefig(pltstr)
            plt.close('all')
       
    if stratg:
        gc = piecewise_model(gammaout, auxg, gvals, extrapolate = extrapolate)
    else:
        gc = np.full((nm),gammaout)
        

    return betac, gammaout, gvals, gc, betaS.squeeze(), gammaS.squeeze()

def piecewise_model(model, auxvar,  vaux, extrapolate = False):
    """
    model is the desired output at the values vaux of the variable auxvar
    """
    
    from scipy import interpolate
    
    auxi = np.copy(auxvar)
    
    #if ~extrapolate:
    #    auxi[auxi >= np.max(vaux)] = np.max(vaux)
    #    auxi[auxi <= np.min(vaux)] = np.min(vaux)

    if extrapolate == False:
        f = interpolate.interp1d(vaux, model, fill_value = (model[0],model[-1]), bounds_error = False)
    else:
        f = interpolate.interp1d(vaux, model, fill_value = 'extrapolate')

    fout = np.squeeze(f(auxi))

    return fout

#%%


def results_to_netcdf(beta1,gamma1,gvals1,gc1,Sbeta1,Sgamma1,stats1,runtag):

    file_out = runtag + 'summary.npy'
    df = {}
    df['beta1'] = beta1
    df['gamma1'] = gamma1
    df['gvals1'] = gvals1
    df['gc1'] = gc1
    df['Sbeta1'] = Sbeta1
    df['Sgamma1'] = Sgamma1
    df['stats1'] = stats1
    np.save(file_out, df, allow_pickle=True)

    
