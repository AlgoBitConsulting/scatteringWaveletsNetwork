
import sys, subprocess
from os import system

sys.path.append('/home/markus/anaconda3/python/modules')

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate
from scipy.stats import multivariate_normal
from scipy.integrate import simps
import copy
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

def prepareRF(nameOfMethod, AL, al, pl, n, silent=False):
   RF         = TD(nameOfMethod)
   RF.AL      = AL
   RF.AL_org  = copy.deepcopy(AL)
   RF.al      = al
   RF.al_org  = copy.deepcopy(al)
   RF.pl      = pl
   RF.pl_org  = copy.deepcopy(pl)
   
   RF.separateDataForRF(len(AL)-n, len(AL))   
   RF.fitRF()
   if not(silent):
      RF.evaluateRF()

   return(RF)

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

   def getAttributeNames(self):
     a     = self.__dir__()
     ac    = a.copy()
     ac.remove('getAttributeNames')
     for b in a:
        if '_' in b:
           ac.remove(b)
     
     return(ac) 



class DD:
   def __init__(self):
      self.description = 'contains all filenames for model data'
      self.pos         = ''
      
class RF:
   def __init__(self, DATA, info):
      self.DATA = DATA
      self.AL   = DATA.AL
      self.al   = DATA.al
      self.pl   = DATA.pl
      self.info = info

   #############################################################################         

   def fitRF(self):
      self.rf = RandomForestClassifier()
      self.rf.fit(self.L1, self.l1)
   
   #############################################################################      

   def separateDataForRF(self, nrCalDat):  
      self.m             = nrCalDat
      self.L1, self.L2   = self.AL[0:nrCalDat],  self.AL[nrCalDat:len(self.AL)] 
      self.l1, self.l2   = self.al[0:nrCalDat],  self.al[nrCalDat:len(self.al)] 
      self.p1, self.p2   = self.pl[0:nrCalDat],  self.pl[nrCalDat:len(self.pl)] 

      print("*** calibration on " + str(nrCalDat) + " elements \n    support size:  " + str( len(self.AL)-nrCalDat) + "\n    total size:    " + str( len(self.AL)))

   #############################################################################      

   def shuffleDataRF(self):
      R  = []
      for ii in range(len(self.AL)):
         R.append([self.AL[ii], self.al[ii], self.pl[ii]])
      np.random.shuffle(R)
      self.AL = list(map(lambda x: x[0], R))
      self.al = list(map(lambda x: x[1], R))
      self.pl = list(map(lambda x: x[2], R))

   #############################################################################      

   def evaluateRF(self, asDict=False):
      y_true = self.l2
      y_pred = self.rf.predict(self.L2)
      erg    = metrics.classification_report(y_true, y_pred, output_dict= asDict)
      if asDict:
         return(erg)
      else:
         print(erg)   
         print("length L1: " + str(len(self.L1)))
         print("length L2: " + str(len(self.L2)))
         return(erg)

   #############################################################################      

   def evaluate(self, n, crossVal = False, dict=False):
      self.shuffleDataRF()
      self.separateDataForRF(n)
      self.fitRF()

      self.erg = self.evaluateRF(asDict=dict)
      if crossVal:
         self.crossVal = '\nCross Validation: ' + str(round(cross_val_score(self.rf, self.L2, self.l2, cv=3).mean(),2 ) )
         print("cross validation: " + str(self.crossVal))            

   #############################################################################            

   def getOverviewOfAnnotations(self):
      al     = self.al   
      aa     = list(set(al))
      aa.append(max(aa)+1)
      M      = np.matrix(np.histogram(al, aa))
      m1     = list(M[0,1].flatten())
      m2     = list(M[0,0].flatten())
      m1.remove(max(m1))
      M      =  mat([m1,m2])
      self.M = M
      return(M)

   #############################################################################      

   def balanceDataForRF(self):

      self.getOverviewOfAnnotations()
      M      = self.M
      ll     = M.tolist()
      el, nr = ll
      mm     = min(nr)

      L      = []
      for ii in range(len(self.AL)):    
         L.append([ self.AL[ii], self.al[ii], self.pl[ii]])

      K = []
      for e in el: 
         k =  list(filter(lambda x: x[1] == e, L))
         k = k[0:mm]
         K.append(k) 
      
      self.K = K 

      self.AL, self.al, self.pl  = [], [], []  
      for k in K:
         AL_k, al_k, pl_k = list(map(lambda x: x[0], k)), list(map(lambda x: x[1], k)), list(map(lambda x: x[2], k))       
         self.AL.extend(AL_k)
         self.al.extend(al_k)
         self.pl.extend(pl_k)

   #############################################################################      

   """
   def normalize(self, v):
      erg = (v - self.mt)/sqrt(self.vt)
      return(erg)   

   def standardize(self, L):
      L1                 = copy.deepcopy(L)
      L1                 = np.matrix(L1)
      v                  = tp(tp(L1).var(1))
      #z                  = np.array(v> vvr , dtype='int')
      #Z                  = np.zeros((L1.shape[1],L1.shape[1])); np.fill_diagonal(Z, z)
      #idx                = np.argwhere(np.all(Z[..., :] == 0, axis=0))
      #Zt                 = np.delete(Z, idx, axis=1)
      #L1t                = L1.dot(Zt)
      M                  = np.tile(tp(tp(L1).mean(1)), (L1.shape[0],1) )
      vt                 = tp(tp(L1).var(1))
     
      V                  = 1/sqrt(np.tile(vt, (L1.shape[0],1) ))
      B                  = np.round( np.array((L1-M))*np.array(V), 2)
      T                  = list(map(list, list(B)))

      mt                 = M[0, :]
      
      return([T, mt, vt])

   def pred_norm_rf(self, C, SWO_2D):
      erg  = self.normalize(np.array(ST.deepScattering(C, SWO_2D)))
      pred = self.rf.predict(erg.tolist())
      return(pred)
   """   
      
    
      
      
###
  

   
    
  
      
 
  
      
      

