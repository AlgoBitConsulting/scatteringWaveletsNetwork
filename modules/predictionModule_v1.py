
import sys, subprocess
from os import system

sys.path.append('/home/markus/anaconda3/python/modules')

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate
from scipy.stats import multivariate_normal
from scipy.integrate import simps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageOps, ImageTk

import timeit
import copy

import DFTForSCN_v7 as DFT
import morletModule_v2 as MM  
import misc_v9 as MISC
import scatteringTransformationModule_2D_v9 as ST


import sklearn
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import csv
import pickle
from datetime import datetime 

import multiprocessing as mp
from multiprocessing import Pool

from joblib import Parallel, delayed
from functools import partial


from pynput import mouse
import tkinter
import pdb

from sys import argv
from sklearn.metrics import auc


pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

#############################################################################         
    

      
#############################################################################           

def mergeWSNC(L,ML, name):
   NOB = MISC.TD(name)
   NOB.L1 = []
   NOB.l1 = []
   NOB.L2 = []
   NOB.l2 = []
   NOB.AL = []
   NOB.al = []
   NOB.pl = []
   
   for ll in range(len(ML)):
      a = L.__getattribute__(ML[ll])
      NOB.L1 = NOB.L1 + a.L1
      NOB.l1 = NOB.l1 + a.l1
      NOB.L2 = NOB.L2 + a.L2
      NOB.l2 = NOB.l2 + a.l2
      NOB.AL = NOB.AL + a.AL
      NOB.al = NOB.al + a.al
      NOB.pl = NOB.pl + a.pl
      
   setattr(L, name, NOB) 
   
   return(L)
   

#############################################################################      

def f(i):
   def fi(x):
      return(x[i])
   return(fi)
     
f0 = f(0)
f1 = f(1)
f2 = f(2)

#############################################################################  

#***
#*** MAIN PART
#***
#
#  exec(open("predictionModule-v1.py").read())
#


class OB:
   def __init__(self, description):     
      self.description = description

class DD:
   def __init__(self):
      self.description = 'contains all filenames for model data'
      self.pos         = ''
      
class TD:
   def __init__(self, name ):
      self.name = name
      self.m = 0
      self.max = 0
      self.DATA = DD()
      self.L1 = []
      self.L2 = []
      self.l1 = []
      self.l2 = []
      self.info = ''

   def loadData(self, path, name, attr=''):
      self.DATA.pos = path + name  
      AL = []
      al = [] 
      pl = []
      CL = [[],[],[]]
      if self.DATA.pos != '':
         nl = list(self.DATA.__dict__.keys())
         nl.remove('description')
         for xx in nl:
            fname = self.DATA.__getattribute__(xx)
            pickle_in   = open(fname,"rb")
            CL          = pickle.load(pickle_in)
            AL          = AL + list(map(f0, CL[0]))
            al          = al + list(map(f1, CL[0]))
            pl          = pl + list(map(f2, CL[0]))
               
      self.AL_org = copy.deepcopy(AL)
      self.AL     = AL
      self.al_org = copy.deepcopy(al)
      self.al = al
      self.pl_org = copy.deepcopy(pl)
      self.pl = pl
      
      self.SWO = CL[1]      


   def prepareRF(self,diff):
      self.separateDataForRF(len(self.AL)-diff, len(self.AL))
      self.rf          = RandomForestClassifier()
      self.rf.fit(self.L1, self.l1)
   
  
   def separateDataForRF(self,m,mmax):  
      L1      = []
      l1      = []
      self.m    = m
      self.mmax = mmax
      for ii in range(self.m):
         L1.append(self.AL[ii])
         l1.append(self.al[ii])

      L2  = []
      l2  = []
      for ii in range(self.m,self.mmax):
         L2.append(self.AL[ii])
         l2.append(self.al[ii])

      self.L1 = L1
      self.l1 = l1
      self.L2 = L2
      self.l2 = l2
    

   def shuffleDataRF(self):
      def f0(x):
         return(x[0])   
      def f1(x):
         return(x[1])
      def f2(x):
         return(x[2])
      
      R  = []
      for ii in range(len(self.AL)):
         R.append([self.AL[ii], self.al[ii], self.pl[ii]])
      np.random.shuffle(R)
  
      self.AL = list(map(f0, R))
      self.al = list(map(f1, R))
      self.pl = list(map(f2, R))


   def evaluateRF(self, asDict=False):
      ypred = self.rf.predict(self.L2)
      erg   = metrics.classification_report(ypred, self.l2, output_dict= asDict)
      if asDict:
         return(erg)
      else:
         print(erg)   
         print("length L1: " + str(len(self.L1)))
         print("length L2: " + str(len(self.L2)))


   def standardize(self, vvr=0.001):
      AL                 = np.matrix(self.AL)
      v                  = tp(tp(AL).var(1))
      z                  = np.array(v> vvr , dtype='int')
      Z                  = np.zeros((AL.shape[1],AL.shape[1])); np.fill_diagonal(Z, z)
      idx                = np.argwhere(np.all(Z[..., :] == 0, axis=0))
      Zt                 = np.delete(Z, idx, axis=1)
      ALt                = AL.dot(Zt)
      M                  = np.tile(tp(tp(ALt).mean(1)), (ALt.shape[0],1) )
      vt                 = tp(tp(ALt).var(1))
      c                  = list(np.array(vt).flatten())
      V                  = 1/sqrt(np.tile(vt, (ALt.shape[0],1) ))
      B                  = np.round( np.array((ALt-M))*np.array(V), 2)
      T                  = list(map(list, list(B)))

      mt                 = M[0, :]
      self.mt            = mt
      self.vt            = vt  
      self.AL            = T
      


   def evaluate(self, n, write = False, crossVal = False):
   
      if type(self.SWO) is MM.SWO_2D:
         print("\nallCoef  : " + str(self.SWO.allCoef))
         if self.SWO.allLevels == False:
            level="only last level"
         else:
            level="all levels"   
         print("allLevels: " + level)
         print("C.shape  : " + str(self.SWO.C.shape))
         print("\n")
     
      if len(self.info)>0:
         print(self.info)
      
      print('##############     ' + self.name + '  ################################')
     
      self.shuffleDataRF()
      self.prepareRF(n)
      self.evaluateRF()
      
      self.standardize()
      
      self.prepareRF(n)
      self.evaluateRF()

      if crossVal:
         print('\nCross Validation: ' + str(round(cross_val_score(self.rf, self.L2, self.l2, cv=3).mean(),2 ) ))

      if write:
         print("write object")
         self = copy.deepcopy(self)   



   def normalize(self, v):
      #try:
      erg = (v - self.mt)/sqrt(self.vt)
      #except:
      #   erg = v
      #   print("no mean and variance found...doing nothing...")
      return(erg)   



   def pred_norm_rf(self, C, SWO_2D):
      erg  = self.normalize(np.array(ST.deepScattering(C, SWO_2D)))
      pred = self.rf.predict(erg.tolist())
      return(pred)
      
      
      
   def getOverviewOfAnnotations(self):
      al = self.al   
      aa = list(set(al))
      aa.append(max(aa)+1)
      M  = np.matrix(np.histogram(al, aa))
      m1 = list(M[0,1].flatten())
      m2 = list(M[0,0].flatten())
      m1.remove(max(m1))
      M =  mat([m1,m2])
      return(M)
      
      
      
   def removeData(self, what):
      AL = []
      al = []
      pl = []
   
      for jj in range(len(self.al)):
         if self.al[jj] not in what:
            AL.append(self.AL[jj])
            al.append(self.al[jj])
            pl.append(self.pl[jj])

      self.AL = AL
      self.al = al
      self.pl = pl
        


   def balanceDataForRF(self, weights):
      aa = list(set(self.al))
      mm = 10000
   
      for ii in range(len(aa)):
         if self.al.count(aa[ii]) < mm:
            mm =  self.al.count(aa[ii])
   
      II = []
      for ii in range(len(aa)):
         zz = 0
         kk = 0
         while zz < len(self.al) and kk < mm*weights[ii]:
            if self.al[zz] == aa[ii]:
               II.append(zz)
               TAO.HTON.pl = TAO.HTON.pl_org

               kk = kk+1
            zz = zz+1
      
      AL = np.array(self.AL)[II]
      al = np.array(self.al)[II]
      pl = np.array(self.pl)[II]
   
      self.II = II
      self.mm = mm
      self.AL = list(AL)
      self.al = list(al)
      self.pl = list(pl)
      
   
   def traMCo_SWC(self, C):
        A = MISC.adaptMatrix(MISC.maxC(C,self.maxC, False), self.adaptMatrix)
        a = np.array(self.normalize(ST.deepScattering(A, self.SWO_2D)))
        
        return(list(a[0, :]))
   
   
   def traMCo_PYW(self, C):   
      erg,c,coeffs   = MISC.usePywt(C, self.wvname, [1,1,1], 1, 3)
      A              = MISC.adaptMatrix(np.array(coeffs[0], dtype='uint8'), self.adaptMatrix)    
      a              = np.array(self.normalize(A.flatten()))
      return(list(a[0,:]))
      
      
   def resetModel(self):
      self.Al = self.HTON.AL_org
      self.al = self.HTON.al_org
      self.pl = self.HTON.pl_org   
      self.evaluate(self.mmax-self.m, True, crossVal = True)
      
      
      
"""
##################################################################################


path         =  "/home/markus/anaconda3/python/data/"     

labels_HTON  = np.loadtxt("/home/markus/anaconda3/python/data/AnnoHTON_train_hochkant.csv", delimiter=',')
labels_HMTOC = np.loadtxt("/home/markus/anaconda3/python/data/AnnoHMTOC_train_hochkant_v2.csv", delimiter=',')

fname1       = "TAO train_hochkantHTON  allCoef=True allLevels=False C.shape=(106, 74) init_eta=2 J=0 kk=1 ll=3 m=2-18.05.2021-11:31:28"

##################################################################################  


n                    = 300
TAO                  = OB("RF-C")
TAO.HTON             = TD("HTON")
TAO.HMTOC            = TD("MTOC")


TAO.HTON.loadData(path, fname1)
SWO                  = TAO.HTON.SWO
TAO.HMTOC            = copy.deepcopy(TAO.HTON)
TAO.HMTOC.al         = labels_HMTOC

withTest             = False
HTON                 = True
HMTOC                = True
HTON_crossVal        = True
HMTOC_crossVal       = True




if HTON:
   TAO.HTON.evaluate(n, HTON_crossVal)   

print('\n\n\n')

if HMTOC:
   TAO.HMTOC.evaluate(n, HMTOC_crossVal)
"""
