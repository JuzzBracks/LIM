import utils2 as ut
import numpy as np
#from astropy import units as u
#import props2 as p
#from importlib import reload
#reload(ut)
#reload(p)

#TIM = reload(p).TIM
nGals = np.array([0.014391114, 0.010868165, 0.0078284155, 0.0060405154])
#specRes = TIM.Instrument.dnu

subSurveys = [ut.GAL_survey(nGal, 250./400.) for nGal in nGals]