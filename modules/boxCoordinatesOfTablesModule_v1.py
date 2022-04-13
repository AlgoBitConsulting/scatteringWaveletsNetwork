import sys, subprocess
from os import system
sys.path.append('/home/markus/anaconda3/python/modules')
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import timeit

import misc_v9 as MISC
import scatteringTransformationModule_2D_v9 as ST
import dataOrganisationModule_v1 as dOM
import morletModule_v2 as MM  

from datetime import datetime 
import pdb
from sys import argv
from copy import deepcopy
from tqdm import tqdm

#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models

import mysql.connector
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()

import sklearn
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import pywt

#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt


############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###########################################################################

def printMat(A):
   print('\n'.join([''.join(['|{:5}'.format(item) for item in row]) for row in A]))

###########################################################################

def evaluateCalibration(p, pred, labels):
   r    = list(range(len(p)))
   erg  = list(1*(pred == labels)) 
   M    = np.array([ list(r), list(p), list(pred), list(erg), list(labels)], dtype='int')
   printMat(tp(M))
   print(erg.count(0)/len(erg))

###########################################################################


def findPos(l,L):
   try:
      pos = L.index(l)
   except:
      pos = -1   

   return(pos) 

###########################################################################

def prepareDataForRF(whichLabelToPredict, SWC, SQL, con):

   rs  = con.execute(SQL)
   HL  = []
   for row in rs:
      rr = list(row)
      for ii in range(len(rr)):
         if rr[ii]==None:
            rr[ii] = ''
      HL.append(rr)
      
   #print(len(HL))
   
   AL = []
   al = [] 
   pl = []
   cc = []

   DATA   = SWC.master.data
   LABELS = SWC.master.labels 
   O      = getattr(LABELS, whichLabelToPredict)

   for ii in range(len(HL)):
      h    = HL[ii][0]
      spos = findPos(h, DATA.HL)
      if spos>=0:
         lpos = findPos(h, O.HL)
         if lpos>=0:
            AL.append(DATA.AL[spos]) 
            al.append(O.LL[lpos])
            pl.append(h)
      else:
         cc.append(h)
   return([AL, al, pl, cc])

###########################################################################

def getColumns(C, noc):
   Dt  = C[150:650, : ]
   D   = np.array(Dt, dtype='uint8')
   erg = []
   
   for ii in range(50, D.shape[1]-50):
      l = D[:, ii].tolist()
      if l.count(255)==D.shape[0]:
         erg.append(ii)
   
   if erg==[]:
      if noc==2:
         dL = [int(Dt.shape[1]/2)]
      if noc==3:
         dL = [int(Dt.shape[1]/3), 2*int(Dt.shape[1]/3)]  
   else:     
      if noc == 3:      
         d = list(np.diff(np.array(erg))>10)
         t = d.index(True)+1
         d1 = np.mean(np.array(erg[0:t]))
         d2 = np.mean(np.array(erg[t:]))
         dL = [int(d1), int(d2)]
      if noc == 2:
         d1 = np.mean(np.array(erg))
         dL = [int(d1)]
      
   return(dL)

###########################################################################

def makeCopiesOfColumns(C, col, noc):

   if (noc not in (2,3)) or (col not in list(range(1, noc+1))):
      return(C)
     
   D = np.ones(C.shape)*255   
   dL = getColumns(C, noc)
   
   if noc==2:
      Cl, Cr        = D.copy(), D.copy()
      
      if col==1:
         Cl[:, :dL[0]] = C[:, :dL[0]]
         Cl_r = np.roll(Cl, dL[0]-30)
         CC   = Cl*Cl_r/255   
      else:
         Cr[:, dL[0]:] = C[:, dL[0]:]
         Cr_l = np.roll( Cr, -(dL[0]-30))
         CC   = Cr*Cr_l/255 
      
   if noc==3:
      shift              = dL[0]
      Cl, Cm,  Cr        = D.copy(), D.copy(), D.copy()
      
      if col ==1:
         Cl[:, :dL[0]]   = C[:, :dL[0]]
         Cl_m            = np.roll(Cl, dL[0]-30)
         Cl_r            = np.roll(Cl, dL[1]-20)
         CC              = Cl*Cl_m*Cl_r/(255**2)
      if col==2:
         Cm[:, dL[0]:dL[1]] = C[:, dL[0]:dL[1]]
         Cm_l               = np.roll(Cm, -(dL[0]-30))
         Cm_r               = np.roll(Cm, dL[0]-30)
         CC                 = Cm*Cm_l*Cm_r/(255**2)
      if col==3:
         Cr[:, dL[1]:]   = C[:, dL[1]:]
         Cr_m            = np.roll(Cr, -(dL[0]-40))
         Cr_l            = np.roll(Cr, -(dL[1]-30))
         CC              = Cr*Cr_m*Cr_l/(255**2)   
 
   return(CC)
 
###########################################################################
 
def makeStripes(Ct, stepSize, windowSize, direction):

   C   = Ct.copy()
   n,m = C.shape
   ii  = 0
   WL  = []
   
   if direction=='H':
      while ii+windowSize < n:   
         W = C[ii:(ii+windowSize), :]
         WL.append([W, ii, ii+windowSize])
         ii = ii+stepSize
   
   if direction=='V':
      while ii+windowSize < m:
         W = C[:, ii:(ii+windowSize)]
         WL.append([W, ii, ii+windowSize])
         ii = ii+stepSize
         
   return(WL) 
    
###########################################################################

def usePywt(C, wvname, w, st, en):

   #print("using " + wvname)
   coeffs    = pywt.wavedec2(C, wvname)
   l         = len(coeffs)
   #print("len coeffs=" + str(l))
   #print("size C         : " + str(C.shape))
   #print("size max coeffs: " + str( coeffs[l-1][0].shape))
   #print("size min coeffs: " + str( coeffs[0].shape))
   #print("\n\n\n")

   nn = 0
   c         = [1*coeffs[0]]  
   for ii in range(1,len(coeffs)):
      b  = np.zeros( (coeffs[ii][0].shape))
      nn = 0
      if ii>= st and ii <= en:
         nn = 1
      c.append([w[0]*nn*coeffs[ii][0], w[1]*nn*coeffs[ii][1], w[2]*nn*coeffs[ii][2]])

   erg = pywt.waverec2(c, wvname)
   return([erg, c, coeffs])        

###########################################################################

def countEntriesMatrix(M):

   L = np.concatenate(M.tolist())
   l = list(set(L))
   l.append( max(l)+1)
   l.sort()
   
   erg = np.histogram(L, l)
   
   return( [ erg[0], l])

###########################################################################

def makeSWCForStripes_V(page, SS_V, MAT, SWO_2D):
   
   CLt_V = list(map(lambda x: x[0], SS_V))
   CL_V  = []

   for cl in CLt_V:
      A = MAT.downSampling(cl, downSamplingRate )
      B = MAT.adaptMatrix(A, adaptMatrixCoef[0], adaptMatrixCoef[1])
      CL_V.append(B)

   ERG_V = MISC.makeIt(CL_V, SWO_2D)
         
   return(ERG_V)

###########################################################################


class boxMaster:
   def __init__(self, name):
      self.name            = name  

###########################################################################

class stripe:
   def __init__(self, C, stepSize, windowSize, direction, SWO_2D, con):
      self.stepSize   = stepSize
      self.windowSize = windowSize
      self.direction  = direction
      self.SWO_2D     = SWO_2D
      self.con        = con
      self.C          = C

   ###########################################################################

   def makeStripes(self, Ct): 
      C   = Ct.copy()
      n,m = C.shape
      ii  = 0
      SS  = []
   
      if self.direction=='H':
         while ii+ self.windowSize < n:   
            W = C[ii:(ii+ self.windowSize), :]
            SS.append([W, ii, ii+self.windowSize])
            ii = ii+self.stepSize
   
      if self.direction=='V':
         while ii+ self.windowSize < m:
            W = C[:, ii:(ii+ self.windowSize)]
            SS.append([W, ii, ii+ self.windowSize])
            ii = ii+ self.stepSize
         
      self.SS = SS
    
   ###########################################################################
 
   def ergForBox(self,ss, box, erg):       # box = ('a1a8bb23ebdbe47718678c34ead7e5137cfb128a0767659d69303075bf49b82f', 209.0, 575.0, 382.0, 663.0)
      
      if box[1] >= self.col_x_min-10 and box[3] <= self.col_x_max+ 10:
         box_x1  = box[1]
         box_x2  = box[3]
         box_y1  = box[2]
         box_y2  = box[4]
         ws      = self.windowSize
         dd      = 0.2
         
         if self.direction == 'H':
            ya      = ss[1]
            ye      = ss[2]            
            if (box_y1 <= ya) and (box_y2 >= ye):  
               erg = 1
            if box_y1>= ya and (box_y1 -ya)/ws <= dd:
               erg = 1 
            if box_y2>= ye and (box_y2 -ye)/ws <= dd:
               erg = 1 
      
         if self.direction == 'V':
            xa  = ss[1]
            xe  = ss[2]
            if (box_x1 <= xa) and (box_x2 >= xe):  
               erg = 1
            if box_x1>= xa and (box_x1 -xa)/ws <= dd:
               erg = 1 
            if box_x2>= xe and (box_x2 -xe)/ws <= dd:
               erg = 1 
                
      return(erg)  
         
   ###########################################################################
      
   def generateLabels(self, hashValue, SS, col):
      SQL    = "select * from TAO where hashValuePNGFile = '" + hashValue + "'"    
      rs     = self.con.execute(SQL)
      COLN   = list(rs.keys())
      l      = list(rs)
      l      = list(l[0])
      page_n = COLN.index('page')
      noc_n  = COLN.index('numberOfColumns')
      page   = l[page_n]
      noc    = l[noc_n]
      WL     = []
      erg    = 0
      
      SQL    = "select * from boxCoordinatesOfTables where hashValuePNG = '" + hashValue + "'"   
      rs     = self.con.execute(SQL)
      K      = list(rs)
      for ss in SS:
         erg    = 0
         for box in K:
            erg = self.ergForBox(ss, box, erg)            

         WL.append([ss[0], [ss[1], ss[2]], hashValue, noc, col, page, erg])

      ergLabel = list(map(lambda x: x[6], WL))

      self.ergLabel = ergLabel
      self.WL  = WL   

   ###########################################################################

   def getColMinMaxCC(self, noc, col):
   
      col_x_max = self.C.shape[1]
      col_x_min = 0     
     
      if col>0:
         if noc>1:
            dL        = getColumns(self.C, noc)
            if dL != []:
               CC        = makeCopiesOfColumns(self.C, col, noc)
               if dL[0]> self.C.shape[1]/2:
                  diff = int( dL[0]- (self.C.shape[1]/2))
                  CC   = np.roll(CC, -diff)
         else:
            CC = self.C
          
         if col ==1 and noc > 1:
            col_x_max = dL[0]
            col_x_min = 0
         if col ==2 and noc ==2:
            col_x_max = C.shape[1]
            col_x_min = dL[0]
         if col ==2 and noc ==3:
            col_x_max = dL[1]
            col_x_min = dL[0]
         if col ==3:
            col_x_max = C.shape[1]
            col_x_min = dL[1]
      else:
         CC = self.C
         print("CC = C")
         
      self.CC        = CC
      self.col_x_min = col_x_min
      self.col_x_max = col_x_max

   ###########################################################################

   def drawErg(self, C, erg):
      WL  = self.WL
      #erg = self.erg
      
      for ii in range(len(WL)):
         wl = WL[ii]
         if erg[ii]==1:
            a,b = wl[1][0], wl[1][1]
            if self.direction == 'H':
               C[a, self.col_x_min: self.col_x_max] = 0
               C[b, self.col_x_min: self.col_x_max] = 0
            if self.direction == 'V':
               if self.col_x_min <= a <= self.col_x_max:
                  C[:, a] = 0
               if self.col_x_min <= b <= self.col_x_max:   
                  C[:, b] = 0
               
      return(C)
    
   ########################################################################### 
     
   def calSWCOfStripes(self):   
      CLt    = list(map(lambda x: x[0], self.WL))
      CL     = []
      MAT    = self.MAT
      for cl in CLt:
         A = MAT.downSampling(cl, self.downSamplingRate )
         B = MAT.adaptMatrix(A, self.adaptMatrixCoef[0], self.adaptMatrixCoef[1])
         CL.append(B)

      self.ERG   = MISC.makeIt(CL, self.SWO_2D)

   ########################################################################### 
   
   def prepareMatrix(self, C):
      A = MAT.downSampling(C, self.downSamplingRate )
      B = MAT.adaptMatrix(A, self.adaptMatrixCoef[0], self.adaptMatrixCoef[1])
      E = MISC.makeIt([B], self.SWO_2D)
      
      return(E)
     
   ###########################################################################    
   
   def prepareData(self, WL):
      CLt = list(map(lambda x: x[0], WL))
      la  = list(map(lambda x: x[6], WL))
      CL  = []

      for cl in CLt:
         A = MAT.downSampling(cl, self.downSamplingRate )
         B = MAT.adaptMatrix(A, self.adaptMatrixCoef[0], self.adaptMatrixCoef[1])
         CL.append(B)
   
      ERG = MISC.makeIt(CL, self.SWO_2D)   
      
      return([ERG, la])
      
   ###########################################################################    
      
   def findStartAndEnd(self,erg):
      foundStart = False
      foundEnd   = False
      start      = 0
      end        = 0
      ii         = 0
      jj         = 0
      boxL       = []
      
      while jj < len(self.WL):
         ii = jj
         while ii < len(self.WL) and not(foundStart):
            if erg[ii] == 1:
               W          = self.WL[ii]
               start      = W[1][0]
               foundStart = True
            ii = ii+1
            
         jj = ii
           
         if foundStart:
            while ii < len(self.WL) and not(foundEnd):
               if np.sum( erg[ii:ii+3]) == 0 or ii>= len(self.WL)-3:
                  W          = self.WL[ii]
                  end        = W[1][1]
                  foundEnd   = True
                  if self.direction =='V':
                     start = min( max(self.col_x_min, start), self.col_x_max)
                     end   = min( max(self.col_x_min, end), self.col_x_max)     
                  boxL.append([start, end])
               ii = ii+1        
         
         jj = ii
         jj = jj+1
         foundStart = False
         foundEnd   = False
            
          
      
      return(boxL)
    
   ###########################################################################  
      
   def correctPrediction(self, correctAt=0.4):
    
      A    = list(map(lambda x: x[1], self.WL))
      prob = self.prob
      erg  = self.erg
      M    = []
       
      for ii in range(len(self.prob)):
         p   = prob[ii]
         a   = A[ii]
         ergalt  = erg[ii]
         if p[1] >= correctAt:
            erg[ii] = 1
         M.append([p[0], p[1], a[0], a[1], erg[ii], ergalt])  
    
      for ii in range(1,len(erg)-1):
         if erg[ii-1] == 0 and erg[ii+1] == 0:
            erg[ii] = 0
         if erg[ii-1] == 1 and erg[ii+1] == 1:
            erg[ii] = 1
      self.erg = erg
    
      return(M)
   
   ###########################################################################   
   
   def prepareMatrixForCNN(self, WL):
      CLt = list(map(lambda x: x[0], WL))
      la  = list(map(lambda x: x[6], WL))
      CL  = []

      for cl in CLt:
         A = MAT.downSampling(cl, self.downSamplingRate )
         B = MAT.adaptMatrix(A, self.adaptMatrixCoef[0], self.adaptMatrixCoef[1])
         CL.append(B)
         
      ML_new = list(np.array(CL)/255)  
      self.train_images, self.train_labels = np.array(ML_new).reshape([len(ML_new), self.adaptMatrixCoef[0],self.adaptMatrixCoef[1],1]), np.array(la)
   
      
   
   ###########################################################################   
    
   def prepMatPywt(self):
   
      CLt    = list(map(lambda x: x[0], self.WL))
      CL     = []
      MAT    = self.MAT
      ERG    = []
      
      for M in CLt:
      
         coeffs = pywt.wavedec2(data=M, wavelet='db2') 
         m      = coeffs[0]
         m      = MAT.adaptMatrix(m, 9,7)
         m      = np.concatenate(m.tolist())
         ERG.append(m.tolist())
   
      self.ERG = ERG
    
   ###########################################################################   
    
   def prepDPywt(self, WL, MAT): 
      CLt = list(map(lambda x: x[0], WL))
      la  = list(map(lambda x: x[6], WL))
      CL  = []
   
      R = tqdm(CLt)
      R.set_description('PYW...')
      for cl in R:
         erg = self.prepMatPywt(cl)
         CL.append(erg)    
     
      return([CL, la]) 
 
   ###########################################################################

   def getBoxCoordinates(self,Ct, DATA, noc, col, display, correctAt=0.4):
   
      if self.direction == 'H' and display.H.check:
         if display.H.SOLL:
            Ct = self.drawErg(Ct, self.ergLabel)  # draw the horizontal box stripes if coordinates for box exists in database
         else:
            if display.H.calcSWC:
               self.calSWCOfStripes()
               self.erg   = DATA.SWC.H.rf.predict(self.ERG)
               self.prob  = DATA.SWC.H.rf.predict_proba(self.ERG)
            
            if display.H.calcPYW:
               self.prepMatPywt()
               self.erg   = DATA.PYW.H.rf.predict(self.ERG)
               self.prob  = DATA.PYW.H.rf.predict_proba(self.ERG)
           
            if display.H.calcCNN:
               self.prepareMatrixForCNN(self.WL)
               self.erg   = model_CNN_H.predict_classes(self.train_images)
               self.prob  = model_CNN_H.predict_proba(self.train_images)
           
            if display.H.correct:
               self.M_H        = self.correctPrediction(correctAt)
               self.N_H        = pd.DataFrame(self.M_H, columns=("prob is no tab","prob is tab","a","b", "corrected: is part of table?", "is part of table?"))   
            
            if display.H.draw:
               Ct         = self.drawErg(Ct, self.erg)
             
      if self.direction == 'V' and display.V.check:
         if display.V.SOLL:
            Ct = self.drawErg(Ct, self.ergLabel)  # draw the vertical box stripes if coordinates for box exists in database
         else:
            if display.V.calcSWC:
               self.calSWCOfStripes()
               self.erg   = DATA.SWC.V.rf.predict(self.ERG)
               self.prob  = DATA.SWC.V.rf.predict_proba(self.ERG)
            
            if display.V.calcPYW:
               self.prepMatPywt()
               self.erg   = DATA.PYW.V.rf.predict(self.ERG)
               self.prob  = DATA.PYW.V.rf.predict_proba(self.ERG)
               
            if display.V.calcCNN:
               self.prepareMatrixForCNN(self.WL)
               self.erg   = model_CNN_V.predict_classes(self.train_images)
               self.prob  = model_CNN_V.predict_proba(self.train_images)   
            
            if display.H.correct:   
               self.M_V        = self.correctPrediction(correctAt)
               self.N_V        = pd.DataFrame(self.M_V, columns=("prob is no tab","prob is tab","a","b", "corrected: is part of table?", "is part of table?"))   
            
            if display.V.draw:
               Ct         = self.drawErg(Ct, self.erg)
         
      return(Ct)

###########################################################################       

def makeOverviewAsMatrix(STPE):
   R   = list(range(len(STPE.erg)))
   M1  = tp(mat( list(STPE.ergLabel)))
   M2  = tp(mat( list(STPE.erg    )))
   P   = np.round(mat(STPE.prob), 2)
   SL  = list(map(lambda x: x[1], STPE.WL))

   M_H = np.concatenate( (M1, P, M2), axis=1)
   return(M_H)
   
###########################################################################   

def rectInRect(r, s, erg=0):
   
   erg   = pointInRect(r, [s[2], s[3]], pointInRect(r, [s[0], s[1]]))
   found = False
   
   if erg==1:
      found = True
      
   return(found)
 
###########################################################################      

def pointInRect(r, point, erg=0):

   x,y= point
   LU_x, LU_y, RL_x, RL_y = r
   
   if LU_x <= x <= RL_x and LU_y <= y <= RL_y:
      erg = 1

   return(erg)
   
###########################################################################

def getNewCoordinates(rect1, rect2):

   r1LU_x, r1LU_y, r1RL_x, r1RL_y = rect1
   r2LU_x, r2LU_y, r2RL_x, r2RL_y = rect2
    
   l = [r1LU_x, r2LU_x, r1RL_x, r2RL_x]
   r = [r1LU_y, r2LU_y, r1RL_y, r2RL_y]
    
   LU_x = min(l)
   LU_y = min(r)
   RL_x = max(l)
   RL_y = max(r)

   return([LU_x, LU_y, RL_x, RL_y] )   
    
###########################################################################

def isEqual(r,s):

   erg    = np.sum(abs(np.array(r)-np.array(s)))
   equal  = False 
   if erg ==0:
      equal = True
      
   return(equal)
      
###########################################################################    
    
def unionOfRectOneRound(rL):    
    
   A = []    
   B = []
   L = rL.copy()
   
   for ii in range(len(rL)):
      r     = rL[ii]
      found = False
      for jj in range(ii+1, len(rL)):
         s = rL[jj]
         if rectInRect(s,r):
            found = True
            rn    = getNewCoordinates(s,r)
            if not rn in A:
               A.append(rn)
               if r in L:
                  L.remove(r)
               if s in L:
                  L.remove(s)
               
      if found == False:
         B.append(r)
    
   return([A,L, B])
   
###########################################################################       
   
def rectUnion(rL):

   L_A, L_L, L_B = [], [], []   
   A = rL.copy()
   found = False
   ii = 0
   while len(A)>0 and ii<= 100:
      a = len(A)   
      A,L, B = unionOfRectOneRound(A) 
      if len(A)<a:
         L_A.append(A)
         L_B.append(B)
         L_L.append(L)
      #else:
      #   found = True 
      ii = ii+1
      
   return([L_A, L_L, L_B])
 
###########################################################################     

def boxesAreParallel(r,s, dist=20):
  
   boxesAreParallel = False
   rLU_x, rLU_y, rRL_x, rRL_y = r
   sLU_x, sLU_y, sRL_x, sRL_y = s
   
   
   if rLU_y == sLU_y and rRL_y == sRL_y and ( (rRL_x > sLU_x) or (sRL_x > rLU_x)) and (abs( rRL_x-sLU_x) < dist or abs( sRL_x - rLU_x)):
      boxesAreParallel = True
    
   if rLU_x == sLU_x and rRL_x == sRL_x and ( (rRL_y > sLU_y) or (sRL_y > rLU_y)) and (abs( rRL_y-sLU_y) < dist or abs( sRL_y-rLU_y) < dist):
      boxesAreParallel = True   
      
   return(boxesAreParallel)   
      
###########################################################################     

def unionOfParallelBoxes(rL, dist):
   A = []    
   B = []
   L = rL.copy()
   
   for ii in range(len(rL)-1):
      r     = rL[ii]
      found = False
      for jj in range(ii+1, len(rL)):
         s = rL[jj]
         if boxesAreParallel(s,r, dist):
            found = True
            rn    = getNewCoordinates(s,r)
            if not rn in A:
               A.append(rn)
               if r in L:
                  L.remove(r)
               if s in L:
                  L.remove(s)
               
      if found == False: # and len(range(ii+1, len(rL)))>0:
         B.append(r)
      if rL[-1] in L:
         B.append(rL[-1])
   
   return([A,L, B])
 
########################################################################### 

def makeUnique(L):

   R = []
   for l in L:
      if not( l in R):
         R.append(l)
    
   return(R)
   
###########################################################################    
 
def drawBox(C, STPE_H, STPE_V):

   box_H  = STPE_H.findStartAndEnd(STPE_H.erg)
   box_V  = STPE_V.findStartAndEnd(STPE_V.erg)
   #C[start_H, : ] = 0
   #C[end_H,   : ] = 0
   #C[:, start_V ] = 0
   #C[:, end_V   ] = 0
   
   img            = Image.fromarray(C)
   draw           = ImageDraw.Draw(img)
   rL             = []
   for jj in range(len(box_V)):
      for ii in range(len(box_H)):
         start_V = box_V[jj][0]
         start_H = box_H[ii][0]
         end_V   = box_V[jj][1]
         end_H   = box_H[ii][1]
         r       = (start_V, start_H, end_V, end_H)
         rL.append(r)
    
   rLt = rL.copy()
      
   if len(rL)>1: 
      L_A, L_L, L_B =rectUnion(rL.copy())
      rLt = np.concatenate(L_B).tolist()
      L_A, L_L, L_B =rectUnion(rLt.copy()) 
      rLt = makeUnique( np.concatenate(L_B).tolist())
      
      if len(rLt)>1:
         A,L,B = unionOfParallelBoxes(rLt, 30)
         rLt   = A+B
           
   for r in rLt:      
      draw.rectangle(r, outline ="red",width=3) 

   return([img, rLt, rL])


class boxBOT:
   def __init__(self, name, SWO_2D, con):
      self.name    = name
      self.SWO_2D  = SWO_2D
      self.con     = con
      
      display                 = boxMaster("master")
      display.H, display.V    = boxMaster("H"), boxMaster("V")

      display.METHOD_H        = 'SWC'
      display.H.calcSWC       = False
      display.H.calcPYW       = False
      display.H.calcCNN       = False

      display.H.check         = True
      display.H.SOLL          = False

      display.H.correct       = True
      display.H.draw          = False

      if display.METHOD_H == 'SWC':
         display.H.calcSWC    = True
      if display.METHOD_H == 'PYW':
         display.H.calcPYW    = True
      if display.METHOD_H == 'CNN':
         display.H.calcCNN    = True


      display.METHOD_V        = 'SWC'
      display.V.draw          = False
      display.V.calcSWC       = False
      display.V.calcPYW       = False
      display.V.calcCNN       = False

      display.V.check         = True
      display.V.SOLL          = False

      display.V.correct       = True

      if display.METHOD_V == 'SWC':
         display.V.calcSWC    = True
      if display.METHOD_V == 'PYW':
         display.V.calcPYW    = True
      if display.METHOD_V == 'CNN':
         display.V.calcCNN    = True
   
      display.drawC           = False
   
      self.display = display
      
      self.MAT                     = dOM.matrixGenerator('downsampling')
      self.MAT.description         = "TEST"
      C                            = self.MAT.generateMatrixFromPNG('/home/markus/anaconda3/python/pngs/train_hochkant/word/train_hochkant-21-portrait-word.png')
            
      self.STPE_H                  = stripe(C, stepSize=5, windowSize=80, direction='H', SWO_2D=self.SWO_2D, con=self.con)
      self.STPE_H.downSamplingRate = 3
      self.STPE_H.adaptMatrixCoef  = tuple([106, 76])
      self.STPE_H.MAT              = self.MAT

      self.STPE_V                  = stripe(C, stepSize=5, windowSize=80, direction='V', SWO_2D=self.SWO_2D, con=self.con)
      self.STPE_V.downSamplingRate = 3
      self.STPE_V.adaptMatrixCoef  = tuple([106, 76])
      self.STPE_V.MAT              = MAT
      
      self.CORRECTION_V       = 0.3
      self.CORRECTION_H       = 0.45

   ###########################################################################

   def makeBoxOnPNG(self, SQL, DATA):
    
      rs                      = self.con.execute(SQL)
      rs                      = list(rs)
      BOXL                    = []
      IMGL                    = []
      CL                      = []
      
      for l in rs:
         hashV                        = l[12]
         C                            = self.MAT.generateMatrixFromPNG(l[1])
         Ct                           = C.copy()
         noc                          = l[6]
         self.STPE_V.hashV            = hashV
         self.STPE_H.hashV            = hashV
         self.STPE_H.C, self.STPE_V.C = C,C
   
         for col in range(1, noc+1):
    
            if self.display.H.check:
               self.STPE_H.getColMinMaxCC(noc, col)
               self.STPE_H.makeStripes(self.STPE_H.CC)   
               self.STPE_H.generateLabels(hashV, self.STPE_H.SS, col)
               Ct  = self.STPE_H.getBoxCoordinates(Ct, DATA, noc, col, self.display, self.CORRECTION_H)
               if not(self.display.H.SOLL):
                  self.M_H = makeOverviewAsMatrix(self.STPE_H)
         
            if self.display.V.check:
               self.STPE_V.getColMinMaxCC(noc, col)
               self.STPE_V.makeStripes(self.STPE_V.CC)   
               self.STPE_V.generateLabels(hashV, self.STPE_V.SS, col)
               Ct  = self.STPE_V.getBoxCoordinates(Ct, DATA, noc, col, self.display, self.CORRECTION_V)
               if not self.display.V.SOLL:
                  self.M_V = makeOverviewAsMatrix(self.STPE_V)
         
            if self.display.H.check and self.display.V.check and not(self.display.H.SOLL or self.display.V.SOLL):
               img,rLt, rL = drawBox(C, self.STPE_H, self.STPE_V)
               if self.display.drawC:
                  draw  = ImageDraw.Draw(img)
                  ss    = "page=" + str(l[4]) + " col=" + str(col) + " H="+ self.display.METHOD_H + " CORRECTION_H=" + str(self.CORRECTION_H) + " V=" + display.METHOD_V + " CORRECTION_V: " + str(self.CORRECTION_V)
                  draw.text( (20,0), ss, font=ImageFont.truetype('Roboto-Bold.ttf', size=15))
               IMGL.append(img)
               BOXL.append(rLt)
               CL.append([self.STPE_H.CC, col, l])
               
            if self.display.H.SOLL or self.display.V.SOLL or self.display.H.draw or self.display.V.draw:
               Image.fromarray(Ct).show()
  
      return([IMGL, BOXL, CL])

########################################################################### 
   
###########################################################################
#***
#*** MAIN PART
#***
#
#  exec(open("TAO-MOD1-getBoxCoordinatesOfTables-v4.py").read())
#

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C                    = MAT.generateMatrixFromPNG('/home/markus/anaconda3/python/pngs/train_hochkant/word/train_hochkant-21-portrait-word.png')
Ct                   = MAT.downSampling(C, 3)                
dx,dy                = 0.15, 0.15
SWO_2D               = MM.SWO_2D(Ct, round(Ct.shape[1]*0.5*dx,3), round(Ct.shape[0]*0.5*dy,3))
SWO_2D.init_eta      = 2
SWO_2D.kk            = 1
SWO_2D.sigma         = mat([[SWO_2D.kk,0],[0,SWO_2D.kk]])
SWO_2D.J             = 0   
SWO_2D.nang          = 8
SWO_2D.ll            = 2
SWO_2D.jmax          = SWO_2D.J
SWO_2D.m             = 2   
SWO_2D.outer         = False
SWO_2D.allCoef       = True
SWO_2D.upScaling     = False
SWO_2D.onlyCoef      = True
SWO_2D.allLevels     = False

############################ CNN and SWC HMTOC ################################################# 

#engine               = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
#con                  = engine.connect()

######################## initialization ###################################

#DATA                 = MISC.loadIt('/home/markus/anaconda3/python/data/DATA-13.09.2021-13:11:22')
   


