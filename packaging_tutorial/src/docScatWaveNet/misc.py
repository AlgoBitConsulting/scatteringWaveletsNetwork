
### standardmäßig in python installiert
import sys, subprocess
from os import system
import os
from PIL import Image, ImageDraw
import pickle
from functools import partial
import timeit
from PIL import Image, ImageDraw, ImageOps, ImageFont
from datetime import datetime 
from copy import deepcopy
import multiprocessing as mp

### eigene Module
"""
sys.path.append('/home/markus/python/scatteringWaveletsNetworks/modules')
sys.path.append('/home/markus/anaconda3/python/development/modules')
import DFTForSCN as DFT
import scatteringTransformationModule as ST
"""

from docScatWaveNet import DFTForSCN as DFT, scatteringTransformationModule as ST



### zu installierende Module
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import tkinter
from joblib import Parallel, delayed
from tqdm import tqdm



#############################################################################  

def makeIt(CL,SWO, des="calculation of SWCs ...", withTime = False):
   
   tt = tqdm(CL)
   tt.set_description_str(des)
   
   t1 = timeit.time.time() 
   foo_ = partial(ST.deepScattering, SWO=SWO)
   output = Parallel(mp.cpu_count())(delayed(foo_)(i) for i in tt)
   if withTime:
      t2 = timeit.time.time(); print(t2-t1)

   return(output)              

#############################################################################  

def saveIt(ERG, fname, withoutDate=False):
   
   a     = datetime.now()
   dstr  = a.strftime("%d.%m.%Y-%H:%M:%S")
   if not(withoutDate): 
      pickle_out = open(fname + '-'+dstr, 'wb')
   else:
      pickle_out = open(fname, 'wb')
   pickle.dump(ERG, pickle_out)
   pickle_out.close()
   return(dstr)

#############################################################################         
        
def loadIt(fname):
   pickle_in   = open(fname,"rb")
   CL          = pickle.load(pickle_in)   
   return(CL)        
          
#############################################################################         

