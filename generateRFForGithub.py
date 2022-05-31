

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

#fname1 = '/home/markus/anaconda3/python/data/TA-bB-H-PNG-18-04-2022-15:47:03'
#fname2 = '/home/markus/anaconda3/python/data/TA-bB-V-PNG-18-04-2022-16:12:09'
#fname3 = '/home/markus/anaconda3/python/data/TA-bBHV-H-PNG-18-04-2022-16:45:52'
#fname4 = '/home/markus/anaconda3/python/data/TA-bBHV-V-PNG-18-04-2022-17:13:58'

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

pathModels = '/home/markus/anaconda3/python/data/'
fname1 = pathModels + 'TA-bB-H-JPG-29.05.2022-00:43:55'
fname2 = pathModels + 'TA-bB-V-JPG-29.05.2022-01:55:49'
fname3 = pathModels + 'TA-bBHV-H-JPG-29.05.2022-03:39:34'
fname4 = pathModels + 'TA-bBHV-V-JPG-29.05.2022-04:50:17'


# workflow: 1) copy /data/fname ---> development/fname 
#           2) fname ---> fname.zst
#           3) rm development/fname
#           4) development/rf/* delete 
#           5) development/fname.zst ---> development/rf/compressed/fname.zst 

class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  

#workingPath     = os.getcwd() + '/alt/'
workingPath      = '/home/markus/anaconda3/python/data/'
NL               = [fname1, fname2, fname3, fname4  ]
SL               = []

for ii in range(len(NL)):
   DATAt     = MISC.loadIt(NL[ii])
   DATA      = boxMaster()
   DATA.rf   = DATAt.rf
   DATA.INFO = DATAt.INFO
   fname     = NL[ii].split('-')[1:4]
   tt        = "TA-"
   for jj in range(len(fname)):
      tt = tt + fname[jj] + '-'
   tt = tt[0:-1]
   SL.append(tt)

   totalName = os.getcwd()  +'/'  + tt
   MISC.saveIt(DATA, totalName, True)
   ss = "zstd " + totalName + " -o " + totalName+ '.zst'
   subprocess.check_output(ss, shell=True, executable='/bin/bash')
   ss = "rm " + totalName
   subprocess.check_output(ss, shell=True, executable='/bin/bash')


workingPath      = os.getcwd() + '/rf/compressed/'
for fl in SL:
   fname = workingPath + fl + '.zst' 
   if os.path.exists(fname): 
      os.remove(fname)

ss = "mv " + os.getcwd() + '/*.zst rf/compressed/'
subprocess.check_output(ss, shell=True, executable='/bin/bash')

