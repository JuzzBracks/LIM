# BSD 3-Clause License
# 
# Copyright (c) 2025 Justin Bracks & Shubh Agrawal
# All rights reserved.

# Inserting TIM2 experimental properties into corresponding xCorr objects.

import utils2 as ut
import numpy as np
from astropy import units as u

TIMdetYield = 0.85 # Assuming 85% yield for TIM KID Detectors.

TIM2 = ut.Telescope(primary = 0.5 * u.m, beamAtten = 'Gaussian', name = 'TIM2')

SW = ut.Instrument(
                    bandWidth = (240 *u.micron, 317 * u.micron),
                    nei = 7.949440193069208 * 1e6 * u.Jy / (u.s ** .5)*2,# *2 is hardcode to account for Diff in Frank and matt's numbers
                    dnu = 4.4 * u.GHz,
                    num_dets = 64 * TIMdetYield,
                    name = 'Short Wavelength 85% Yield')
LW = ut.Instrument(
                   bandWidth = (317 *u.micron, 420 * u.micron),
                   nei = 3.194133716130952 * 1e6 * u.Jy / (u.s ** .5)*2,# *2 is hardcode to account for Diff in Frank and matt's numbers
                   dnu = 3.3 * u.GHz,
                   num_dets = 51 * TIMdetYield,
                   name = 'Long Wavelength 85% Yield')

CII = ut.EmissionLine(158 *u.micron) #TODO Add Emission line file separately and load
#this in from there. Maybe build in the ability for the eLines to carry model values.

instrs = [SW,SW,LW,LW]
binEdges = np.unique(np.concatenate((SW.get_band_edges(2), LW.get_band_edges(2))))
zBins = [ut.zBin(TIM2, instrs[i], CII, binEdges[i], binEdges[i+1]) for i in range(len(binEdges)-1)]

subSurveys = [ut.LIM_survey(TIM2, zBin.Instrument, zBin, np.sqrt(2)*u.deg, np.sqrt(2)*u.deg, 24*u.hr.to(u.s))
                         for zBin in zBins]
