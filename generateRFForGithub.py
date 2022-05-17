

from docScatWaveNet import dataOrganisationModule as dOM, misc as MISC, morletModule as MM, scatteringTransformationModule as ST, tableFinder as TF, DFTForSCN as DFT
import os, numpy as np
import sys, subprocess, os,glob
from PIL import Image, ImageDraw, ImageOps, ImageFont

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real


#***
#*** MAIN PART
#***
#
#  exec(open("generateRFForGithub.py").read())
#
##   

fname1 = '/home/markus/anaconda3/python/data/TA-bB-H-PNG-18-04-2022-15:47:03'
fname2 = '/home/markus/anaconda3/python/data/TA-bB-V-PNG-18-04-2022-16:12:09'
fname3 = '/home/markus/anaconda3/python/data/TA-bBHV-H-PNG-18-04-2022-16:45:52'
fname4 = '/home/markus/anaconda3/python/data/TA-bBHV-V-PNG-18-04-2022-17:13:58'

#fname1 = '/home/markus/anaconda3/python/data/TA-bB-H-JPG-12.05.2022-12:58:18'
#fname2 = '/home/markus/anaconda3/python/data/TA-bB-V-JPG-12.05.2022-13:20:45'
#fname3 = '/home/markus/anaconda3/python/data/TA-bBHV-H-JPG-12.05.2022-13:38:33'
#fname4 = '/home/markus/anaconda3/python/data/TA-bBHV-V-JPG-12.05.2022-14:04:35'


#fname1 = '/home/markus/anaconda3/python/data/TA-bB-H-JPG-16.05.2022-00:24:53'
#fname2 = '/home/markus/anaconda3/python/data/TA-bB-V-JPG-16.05.2022-00:51:15'
#fname3 = '/home/markus/anaconda3/python/data/TA-bBHV-H-JPG-16.05.2022-01:07:20'
#fname4 = '/home/markus/anaconda3/python/data/TA-bBHV-V-JPG-16.05.2022-01:22:35'


#fname1 = '/home/markus/anaconda3/python/data/TA-bB-H-JPG-17.05.2022-00:35:24'
#fname2 = '/home/markus/anaconda3/python/data/TA-bB-V-JPG-17.05.2022-01:17:19'
#fname3 = '/home/markus/anaconda3/python/data/TA-bBHV-H-JPG-17.05.2022-02:14:45'
#fname4 = '/home/markus/anaconda3/python/data/TA-bBHV-V-JPG-17.05.2022-02:55:41'



class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  

#workingPath      = os.getcwd() + '/alt/'
workingPath      = '/home/markus/anaconda3/python/data/'

FL               = ['TA-bB-H-JPG'   , 'TA-bB-V-JPG'   ,  'TA-bBHV-H-JPG',  'TA-bBHV-V-JPG'  ]
DL               = ['-16.05.2022-00:24:53' , '-16.05.2022-00:51:15' ,  '-16.05.2022-01:07:20',  '-16.05.2022-01:22:35'  ]
for ii in range(len(FL)):
   DATAt     = MISC.loadIt(workingPath + FL[ii] + DL[ii])
   DATA      = boxMaster()
   DATA.rf   = DATAt.rf
   DATA.INFO = DATAt.INFO
   MISC.saveIt(DATA, os.getcwd()  +FL[ii], True)
   ss = "zstd " + os.getcwd()  + FL[ii] + " -o " + FL[ii].replace('-onlyTA', '')+ '.zst'
   subprocess.check_output(ss, shell=True, executable='/bin/bash')

workingPath      = os.getcwd() + '/rf/compressed/'
for fl in FL:
   fname = workingPath + fl + '.zst' 
   if os.path.exists(fname): 
      os.remove(fname)

ss = "mv " + os.getcwd() + '/*.zst rf/compressed/'
subprocess.check_output(ss, shell=True, executable='/bin/bash')
