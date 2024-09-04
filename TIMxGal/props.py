# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal
# All rights reserved.

# writing down survey properties

from astropy import units as u, constants as c
import utils
from obj import AttrDict
from astropy.cosmology import Planck18 as cosmo
import numpy as np

TIM = AttrDict()
TIM.det_yield = 0.85 #Percent of detectors that are actually useable. 
TIM.time = (200*u.hr).to(u.s)
TIM.window = True
TIM.useshot = True
TIM.mirror = 2 * u.m
TIM.Daz = 0.2 * u.deg
TIM.Del = 1 * u.deg
TIM.line = 157.74 * u.micron # Cooksy et al. 1986, [CII]


TIM.SW = AttrDict()
TIM.LW = AttrDict()

TIM.SW.min = 240 * u.micron
TIM.SW.max = 317 * u.micron
TIM.SW.NEI = 12.41e7 * u.Jy / (u.s ** .5)
TIM.SW.num_dets = 64 * TIM.det_yield
TIM.SW.dnu = 4.4 * u.GHz

TIM.LW.min = 317 * u.micron
TIM.LW.max = 420 * u.micron
TIM.LW.NEI = 6.81e7 * u.Jy / (u.s ** .5)
TIM.LW.num_dets = 51 * TIM.det_yield
TIM.LW.dnu = 3.3 * u.GHz


for band in ['SW', 'LW']:
    TIM[band].cen = (TIM[band].min + TIM[band].max) / 2
    TIM[band].FWHM = (1.22 * TIM[band].cen / TIM.mirror).to("").value * u.rad
    TIM[band].zmin = utils.l2z_CII(TIM[band].min)
    TIM[band].zcen = utils.l2z_CII(TIM[band].cen)
    TIM[band].zmax = utils.l2z_CII(TIM[band].max)
    TIM[band].ins = utils.Instrument(TIM[band].NEI, TIM[band].FWHM, \
        TIM[band].dnu, TIM[band].num_dets)

AstroDeep = AttrDict()
# TIM bins
#AstroDeep.n_gals = np.array([0.005, 0.002, 0.0019713, 0.000836]) / cosmo.h ** 3
AstroDeep.n_gals = (np.array([0.005, 0.002, 0.0019713, 0.000836]) / cosmo.h ** 3)

#AstroDeep.n_gals = np.array([0.033137304142686355, 0.01039149916371776, 0.010016817295520536,0.005243669958697198])

Euclid = AttrDict()
# TIM bins
Euclid.n_gals = np.array([0.0144, 0.01077, 0.0081,0.0056]) #little h included in these numbers.


                            #TIM 2
#---------------------------------------------------------------------------------------------------------------------
TIM2 = AttrDict()
TIM2.det_yield = 1.7 #Percent of detectors that are actually useable. 
TIM2.time = (100*u.hr).to(u.s)
TIM2.window = True
TIM2.useshot = True
TIM2.mirror = 0.5 * u.m
TIM2.Daz = 3 * u.deg
TIM2.Del = 3 * u.deg
TIM2.line = 157.74 * u.micron # Cooksy et al. 1986, [CII]


TIM2.SW = AttrDict()
TIM2.LW = AttrDict()

TIM2.SW.min = 240 * u.micron
TIM2.SW.max = 317 * u.micron
TIM2.SW.NEI = 1191067 * u.Jy / (u.s ** .5)
TIM2.SW.num_dets = 64 * TIM2.det_yield
TIM2.SW.dnu = 4.4 * u.GHz

TIM2.LW.min = 317 * u.micron
TIM2.LW.max = 420 * u.micron
TIM2.LW.NEI = 2382794 * u.Jy / (u.s ** .5)
TIM2.LW.num_dets = 51 * TIM2.det_yield
TIM2.LW.dnu = 3.3 * u.GHz


for band in ['SW', 'LW']:
    TIM2[band].cen = (TIM2[band].min + TIM2[band].max) / 2
    TIM2[band].FWHM = (1.22 * TIM2[band].cen / TIM2.mirror).to("").value * u.rad
    TIM2[band].zmin = utils.l2z_CII(TIM2[band].min)
    TIM2[band].zcen = utils.l2z_CII(TIM2[band].cen)
    TIM2[band].zmax = utils.l2z_CII(TIM2[band].max)
    TIM2[band].ins = utils.Instrument(TIM2[band].NEI, TIM2[band].FWHM, \
        TIM2[band].dnu, TIM2[band].num_dets)
    #----------------------------------------------------------------------------------------------------
    
                            #SPACE TIM
#----------------------------------------------------------------------------------------------------   
SpaceTIM = AttrDict()
SpaceTIM.det_yield = 1.6 #Percent of detectors that are actually useable. 
SpaceTIM.time = (1000*u.hr).to(u.s)
SpaceTIM.window = True
SpaceTIM.useshot = True
SpaceTIM.mirror = 0.5 * u.m
SpaceTIM.Daz = 3 * u.deg
SpaceTIM.Del = 3 * u.deg
SpaceTIM.line = 157.74 * u.micron # Cooksy et al. 1986, [CII]


SpaceTIM.SW = AttrDict()
SpaceTIM.LW = AttrDict()

SpaceTIM.SW.min = 240 * u.micron
SpaceTIM.SW.max = 317 * u.micron
SpaceTIM.SW.NEI = 1191067.084857143 * u.Jy / (u.s ** .5)
SpaceTIM.SW.num_dets = 64 * SpaceTIM.det_yield
SpaceTIM.SW.dnu = 4.4 * u.GHz

SpaceTIM.LW.min = 317 * u.micron
SpaceTIM.LW.max = 420 * u.micron
SpaceTIM.LW.NEI = 2382794.141587301 * u.Jy / (u.s ** .5)
SpaceTIM.LW.num_dets = 51 * SpaceTIM.det_yield
SpaceTIM.LW.dnu = 3.3 * u.GHz


for band in ['SW', 'LW']:
    SpaceTIM[band].cen = (SpaceTIM[band].min + SpaceTIM[band].max) / 2
    SpaceTIM[band].FWHM = (1.22 * SpaceTIM[band].cen / SpaceTIM.mirror).to("").value * u.rad
    SpaceTIM[band].zmin = utils.l2z_CII(SpaceTIM[band].min)
    SpaceTIM[band].zcen = utils.l2z_CII(SpaceTIM[band].cen)
    SpaceTIM[band].zmax = utils.l2z_CII(SpaceTIM[band].max)
    SpaceTIM[band].ins = utils.Instrument(SpaceTIM[band].NEI, SpaceTIM[band].FWHM, \
        SpaceTIM[band].dnu, SpaceTIM[band].num_dets)