import sys, subprocess
from os import system
sys.path.append('/home/markus/anaconda3/python/development/modules')
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import timeit

from bs4 import BeautifulSoup

import misc_v9 as MISC
import scatteringTransformationModule_2D_v9 as ST
import dataOrganisationModule_v3 as dOM
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
import matplotlib.pyplot as plt


############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###########################################################################

class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  


###########################################################################   

###################        getting boxes               ########################################
###################                                    ########################################
###################           begin                    ########################################

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

def boxesAreParallel(r,s, dist=20):
  
   boxesAreParallel = False
   rLU_x, rLU_y, rRL_x, rRL_y = r
   sLU_x, sLU_y, sRL_x, sRL_y = s
   
   
   if rLU_y == sLU_y and rRL_y == sRL_y and ( (rRL_x > sLU_x) or (sRL_x > rLU_x)) and (abs( rRL_x-sLU_x) < dist or abs( sRL_x - rLU_x) < dist):
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

def pointInRect(r, point, erg=0):

   x,y= point
   LU_x, LU_y, RL_x, RL_y = r
   
   if LU_x <= x <= RL_x and LU_y <= y <= RL_y:
      erg = 1

   return(erg)
   
###########################################################################

def rectInRect(r, s, erg=0):
   
   erg   = pointInRect(r, [s[2], s[3]], pointInRect(r, [s[0], s[1]]))
   found = False
   
   if erg==1:
      found = True
      
   return(found)
 
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

def getBoxes(box_H, box_V):

   rL             = []
   for jj in range(len(box_V)):
      for ii in range(len(box_H)):
         start_V = box_V[jj][0]
         start_H = box_H[ii][0]
         end_V   = box_V[jj][1]
         end_H   = box_H[ii][1]
         r       = (start_V, start_H, end_V, end_H)
         if abs(start_V-end_V)>5 and abs(start_H-end_H)> 5:
            rL.append(r)
            
   rLn = rL.copy()
   for jj in range(len(rL)):
      box1           = rL[jj]
      x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3] 
      for ii in range(jj, len(rL)):
         box2    = rL[ii]
         v1, w1, v2, w2 = box2[0], box2[1], box2[2], box2[3] 
         if y1==w1 and y2==w2 and v1 <= x2 and box1!=box2:
            if box1 in rLn:
               rLn.remove(box1)
            if box2 in rLn:
               rLn.remove(box2)
            r = (x1, y1, v2, w2)            
            rLn.append(r)
            
   rLt = rL.copy()
   if len(rL)>1: 
      L_A, L_L, L_B =rectUnion(rL.copy())
      rLt = np.concatenate(L_B).tolist()
      L_A, L_L, L_B =rectUnion(rLt.copy()) 
      rLt = makeUnique( np.concatenate(L_B).tolist())
      
      if len(rLt)>1:
         A,L,B = unionOfParallelBoxes(rLt, 10)
         rLt   = A+B
   
   return([rLt, rL, rLn])

################################################################################################   


###################        getting boxes               ########################################
###################                                    ########################################
###################           end                      ########################################



###################  getting column line                ########################################
###################                                     ########################################
###################      begin                          #######################################

def removeColsWithEntriesInNeighbour(LM, MK):
   for jj in range(len(MK)):
      r = MK[jj]
      for kk in range(len(LM)):
         lm  = LM[kk][0]
         lmt = lm.copy()
         for zz in range(len(lm)):
            s = lm[zz]
            if s[0] <= r <= s[2]:
               lmt.remove(s)
               LM[kk][0] = lmt

   return(LM)

################################################################################################

def removeEmptyColsFromMK(LM,MK):

   for jj in range(len(LM)):
      lm = LM[jj]
      if len(lm[0])==0:
         MK.remove(lm[1][0])

   return(MK)

################################################################################################

def sortInCols( KxL, midL):
    LM                    = []
    PKxL                  = KxL.copy()
    for kk in range(1,len(midL)):
       lm = []
       zz = 0
       for jj in range(len(KxL)):
          x1,y2,x2,y1 = list( map(lambda x: round(x), KxL[jj]))  # (x1,y1,x2,y2)
          if midL[kk-1] <= x1 <= midL[kk]:
             if KxL[jj] in PKxL:
                #if x2 > midL[kk]:
                #   zz = zz+1
                lm.append(KxL[jj])
                PKxL.remove(KxL[jj])
       #if zz < len(lm):
       LM.append([lm, [midL[kk-1], midL[kk]]])

    return([LM, PKxL])

################################################################################################

def getColumnsOfTable(C, a, box, jump=2, d =1):
   b = np.sign(np.diff(a))
   l = np.where(b<0)[0]
   R = []

   for ii in range(0,len(l)-1):
      start = l[ii]
      end = l[ii+1]
      jj = end
      found = False
      while jj > start and b[jj]<=0: 
         jj = jj-1 
     
      if jj > start:
         found = True
     
      if found:
         r = np.max(a[jj:end+1])
         R.append([r, [jj,end]])
               
   pR = []     
   for ii in range(len(R)):
      if R[ii][0]>= jump:
         pR.append(R[ii])
         r = R[ii]
         s = r[1]
         p = int( 0.5*(s[0]+s[1]))
         C[:, p-d:p+d] = 0         
         
   return([C, R, pR, b, l])

################################################################################################ 

def MA(R, n):
   a = []
   for ii in range( n, len(R)-n-1):
      r= R[ii-n:ii]
      a.append(np.mean(r))   
   return(a)

################################################################################################  
   
def countC(C, what):
  
   erg = []

   for ii in range(C.shape[1]):
      l = list(C[:, ii])
      e = l.count(what)
      erg.append(e)

   return(erg)

################################################################################################ 

def moveBorder(C, y, direction='D'):
   global white
   s =-1
   if direction=='D':
      s = 1
      
   k  = list(C[y, :])
   zz = 1
   found = False
   while not(found) and zz <= 15:
      k  = list(C[y+s*zz, :])
      if k.count(white) == len(k):
         found = True
         yn = y+s*zz
      zz = zz+1 
   erg = y
   if found:
      erg = yn
   
   return(erg)

################################################################################################  

def mP(box, C,n, searchWhat, plotMA=False):

   y1,  y2, x1, x2  = box[1], box[3], box[0], box[2]
   y1n, y2n         = moveBorder(C, y1), moveBorder(C, y2, 'U')
   boxn             = [x1, y1n, x2, y2n]
   Ct               = C[y1n:y2n,x1:x2]
   erg              = countC(Ct, searchWhat)
   y                = MA(erg, n)

   return([erg, y, boxn, Ct])

################################################################################################

def restrictBoxesJPG(BL, box, colmax, colmin, tol=0.7, p=0.15):

   y1, y2, x1, x2   = box[1], box[3], box[0], box[2]
   K                = []

   for ii in range(len(BL)):
      (bx1, by1, bx2, by2, btx) = BL[ii]
      if ( colmin <= bx1 <= bx2 <= colmax) and ( x1 <= bx1 <= bx2<= x2) and ( y1 <= by1 <= by2<= y2):
         K.append([ [ bx1, by1, bx2, by2],btx,(bx2-bx1)/(colmax-colmin) ])
   
   return(K)

###########################################################################

def allColumnsOfTables(C2, img, rL, GEN, page, xmm, white=255):

   print("calculating column lines...")
   GEN.outputFolder  = "/home/markus/anaconda3/python/"
   GEN.outputFile    = "MOD1"
   BL                = GEN.groupingInLine(page)
   C                 = np.array(img).copy()
   LLL               = []
   KKK               = []
   MMM               = []
   RRR               = []
   PPP               = []
   ERG               = []

   for ii in range(len(rL)):
      boxO, col             = rL[ii]
      K                     = restrictBoxesJPG(BL, boxO, xmm[col][1], xmm[col][0], tol=0.7, p=0.15)
      erg, erg_MA, box, Ct  = mP(boxO, C2.copy(), 10, white, False)     
      bx1, by1, bx2, by2    = box[0], box[1], box[2], box[3]
      Ct                    = C2[by1:by2, bx1:bx2]      
      jump                  = 0.7*(by2-by1)
      Ct, R, pR, b, ll      = getColumnsOfTable(Ct, erg, box, jump=jump, d=1) 

      midL                  = [0] + list(map(lambda x: round(sum(x[1])*0.5), pR.copy())) + [bx2-bx1]
      midL                  = list(np.array(midL) + 1*bx1)
      KxL                   = list(map(lambda x: x[0], K))
               
      LM, pKxL              = sortInCols( KxL, midL)
      MK                    = removeEmptyColsFromMK(LM, midL.copy())            
      LM, pKxL              = sortInCols( KxL, MK)
      LM                    = removeColsWithEntriesInNeighbour(LM, MK) 
      MK                    = removeEmptyColsFromMK(LM, MK)  
               
      LMT = []
      for jj in range(len(LM)):
         if len(LM[jj][0])>0:
            LMT.append(LM[jj])
               
      W = []      
      for jj in range(len(LMT)):
         x,y = LMT[jj][1]      
         C[by1:by2, y-1:y+1] = 0
                           
      ERG.append([rL[ii], LMT])

   print("done...")

   return([C, ERG])

###################     getting column line            ########################################
###################                                    ########################################
###################           end                      ########################################


###########################################################################

def flattenP(p):

   q = p[0:2]
   for ii in range(2,len(p)-3):
      r = p[ii-2:ii+3]       
      q.append(np.median(r))
   q.extend([p[-3],p[-2], p[-1]])

   p, q = np.round(p,2), np.round(q,2)
   
   return([p, q])

############################################################################

def getBoxCoordinatesUsingRF(rf, ERG, flatten=False):

   erg, M    = rf.predict(ERG), rf.predict_proba(ERG)

   if flatten:
      p,q       = flattenP( list(M[:,1]))
      M[:,1]    = q
   
   M = np.round(M,2)
   return([M, erg])

###########################################################################

def decomposeMatrices(DATA, STPE, INFO):

   print("calculating horizontal and vertivcal decompositions ...") 
   KBOX = INFO.KBOX.copy()

   if INFO.copyHL:  # die Zerlegungen für HL sind i.A. identisch mit TA
      try:
         KBOX.remove('HL')
      except:
         a=3

   for kindOfBox in KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX: 

            INFO.kindOfBox  = kindOfBox
            INFO.method     = method 
            INFO.direction  = direction
            
            O               = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)
            STPE.direction  = direction
            STPE.windowSize = O.windowSize
            STPE.stepSize   = O.stepSize

            INFO            = decomposeMatricesForLeaf( DATA, STPE, INFO)   
          
   return(INFO) 

###########################################################################

def decomposeMatricesForLeaf(DATA, STPE, INFO):
  
   CL, xmm  = DATA.CL, DATA.xmm
   if INFO.method == 'bBHV':
      CL = DATA.CL_bbm

   O  = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)

   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)

   for ii in range(len(CL)):
      l                  = boxMaster("") 
      C, col, noc        = CL[ii]
      l.WL, _, _         = STPE.makeWL([CL[ii]], [], xmm, page, 'noHashValueAvailable')
      l.ii, l.col, l.noc = ii, col, noc
     
      setattr( O, 'WL' + str(ii), l)
      if INFO.copyHL:
         setattr(OC, 'WL' + str(ii), l)

   return(INFO)

###########################################################################

def calculateSWCs(DATA, STPE, INFO, des=''):

   print("calculating horizontal and vertical SWCs ") 
   KBOX = INFO.KBOX.copy()

   if INFO.copyHL:  # die Zerlegungen für HL sind i.A. identisch mit TA und damit auch die SWC
      try:
         KBOX.remove('HL')
      except:
         a=3
   
   for kindOfBox in KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX:

            INFO.kindOfBox        = kindOfBox
            INFO.method           = method 
            INFO.direction        = direction

            O                     = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction) 
            STPE.direction        = direction
            STPE.windowSize       = O.windowSize
            STPE.stepSize         = O.stepSize
            STPE.downSamplingRate = O.downSamplingRate
            STPE.adaptMatrixCoef  = O.adaptMatrixCoef

            INFO                  = calculateSWCsForLeaf( DATA, STPE, INFO, des='calculate SWC for '+ kindOfBox + '-' + method + '-' + direction)   

   return(INFO) 

###########################################################################

def calculateSWCsForLeaf(DATA, STPE, INFO, des=''):

   O  = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)
 
   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)

   for ii in range(len(DATA.CL)):

      nameWL   = 'WL'+str(ii)
      WL       = getattr(O, nameWL).WL
      nameAL   = 'AL'+str(ii)
      ss       = 'horizontal rf '+ des + ' column ' + str(ii)
      if INFO.direction == 'V':
         ss = 'vertical rf '+ des + ' column ' + str(ii)

      AL, _   = STPE.prepareData(WL, ss) 
      
      setattr( O, nameAL, AL)
      if INFO.copyHL:
         setattr(OC, 'AL' + str(ii), AL)

   return(INFO)

###########################################################################

def applyRF(DATA, INFO, des=''):

   print("calculation predictions...")

   for kindOfBox in INFO.KBOX:
      for method in INFO.MBOX: 
         for direction in INFO.DBOX: 
            INFO.kindOfBox = kindOfBox
            INFO.method    = method 
            INFO.direction = direction
            INFO           = applyRFForLeaf(DATA, INFO)  

   return(INFO) 

###########################################################################

def applyRFForLeaf(DATA, INFO):

   flatten = INFO.flatten
   O  = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.direction)
  
   if INFO.copyHL:
      OC = getattr( getattr( getattr(INFO,  'HL'), INFO.method), INFO.direction)
 
   for ii in range(len(DATA.CL)):
      nameAL  = 'AL' + str(ii)
      AL      = getattr(O, nameAL)
      M, erg  = getBoxCoordinatesUsingRF(O.rf, AL,flatten)
      
      setattr(O, 'M'+ str(ii), M) 
      setattr(O, 'erg' + str(ii), erg)

   return(INFO)

###########################################################################

def findStartAndEnd(erg, WL,lenErg=3, stin=0, enin=1):

      foundStart = False
      foundEnd   = False
      start      = 0
      end        = 0
      ii         = 0
      jj         = 0
      boxL       = []
      
      try:
         windowSize = WL[0][1][1] - WL[0][1][0]
         stepSize   = WL[1][1][1] - WL[0][1][1]
      except:
         print("findStartAndEnd: taking default values for windowSize and stepSize...")
         windowSize = 30
         stepSize   = 5

      while jj < len(WL):
         ii = jj
         while ii < len(WL) and not(foundStart):
            if erg[ii] == 1:
               W          = WL[ii]
               start      = W[1][0]  + int(0.5*(windowSize- 2*stepSize)) 
               foundStart = True
            ii = ii+1
         jj = ii
           
         if foundStart:
            while ii < len(WL) and not(foundEnd):
               if np.sum( erg[ii:ii+lenErg]) == 0: 
                  W          = WL[ii-1]
                  end        = W[1][1]  - int(0.5*(windowSize- 2*stepSize)) 
                  foundEnd   = True
                  boxL.append([start, end])
               else:
                  if ii>= len(WL)-3:
                     W          = WL[-1]
                     end        = W[1][1] 
                     foundEnd   = True
                     boxL.append([start, end])
               ii = ii+1               
         jj = ii
         jj = jj+1
         foundStart = False
         foundEnd   = False
            
      return(boxL)

###########################################################################

def correctMatrix(M, bb):

   N = tp(M).tolist()
   a = list( np.array( M[:, 2] >= bb, dtype='int')) 
   N.append(a)

   N = tp(np.matrix(N))        

   return(N)

###########################################################################

def valuationMatrix(INFO, lCL):
   
   O1          = getattr(INFO,  INFO.kindOfBox) 
   P_bB        = getattr( getattr( getattr(INFO,  INFO.kindOfBox), 'bB'),   INFO.direction)
   P_bBHV      = getattr( getattr( getattr(INFO,  INFO.kindOfBox), 'bBHV'), INFO.direction)

   correction  = getattr(O1, 'correction-'+ INFO.direction)
   bBHV_weight = getattr(O1, 'weightbBHV-'+ INFO.direction)
   bB_weight   = getattr(O1, 'weightbB-'+ INFO.direction)

   for ii in range(lCL):    
      erg_bB, erg_bBHV = getattr( P_bB, 'erg'+ str(ii)), getattr( P_bBHV, 'erg'+ str(ii))
      M_bB  , M_bBHV   = getattr( P_bB, 'M'+ str(ii)), getattr( P_bBHV, 'M'+ str(ii))
      erg              = 1*( bB_weight*np.array( erg_bB) + bBHV_weight*np.array(erg_bBHV))
      prob             = 1*( bB_weight*M_bB[:,1] + bBHV_weight*M_bBHV[:,1])
      rH               = list( range(len(erg)))
      Mt               = tp( [rH, erg, prob])
      M                = correctMatrix(Mt, correction)
   
      setattr(O1, 'M'+ INFO.direction + str(ii), M)

   return(INFO)

############################################################################

def shiftBoxes(boxLn, m, a,b, size=(840, 596)):

   boxL = boxLn.copy()
   l = int(0.5*(m- (b-a)))
   d = a-l
   for ii in range(len(boxL)):
      x1,y1,x2,y2 = boxL[ii]
      boxL[ii]    = max(a+5, x1+d), y1, min( x2+d, b-5, size[1]-20), y2

   return(boxL)
 
############################################################################  

def allBoxes(kindOfBox, noc, INFO, xmm, m, lenErg=3): 
   
   #rLN_TA = shiftBoxes(rLN_TA, m, a,b)
   INFO.kindOfBox   = kindOfBox
   INFO.direction   = 'H'
   INFO             = valuationMatrix(INFO, len(DATA.CL)) 
   INFO.direction   = 'V'
   INFO             = valuationMatrix(INFO, len(DATA.CL)) 

   OM               = getattr(INFO, kindOfBox)
   OH               = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'H')     ## wird nur für Ermittlung von WL benutzt
   OV               = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'V')     ## wird nur für Ermittlung von WL benutzt
   BOXES            = []

   for col in range(noc):
      M_H, M_V       = getattr(OM,  'MH'+ str(col)), getattr( OM, 'MV'+ str(col))
      WLH, WLV       = getattr(OH, 'WL'+ str(col)).WL, getattr(OV, 'WL'+ str(col)).WL
      boxL_H, boxL_V = findStartAndEnd(M_H[:, 3], WLH, lenErg), findStartAndEnd(M_V[:, 3], WLV, lenErg)   
      rLt, rL, rLn   = getBoxes(boxL_H, boxL_V)
     
      a,b            = xmm[col][0], xmm[col][1]         
      rLn            = shiftBoxes(rLn, m, a, b)

      BB             = list(map( lambda x: [x, col], rLn))
      BOXES.extend(BB)
   
   return(BOXES)

############################################################################ 

def makeImage(Corg):

   n,m                          = Corg.shape
   imgCol                       = Image.new(mode="RGB",size=(m, n), color=(255,255,255))
   A                            = np.array(imgCol)
   A[:,:,0], A[:,:,1], A[:,:,2] = Corg, Corg, Corg   
   img                          = Image.fromarray(A)         
   draw                         = ImageDraw.Draw(img)

   return([img, draw])

############################################################################ 

def filterHL(rLN_TA, rLN_HL, d=20):

   ERG = []
   for ii in range(len(rLN_HL)):
      x1,y1,x2,y2 = boxHL = rLN_HL[ii][0]
      for jj in range(len(rLN_TA)):
         v1, w1, v2, w2 = boxTA = rLN_TA[jj][0]
         if y1 <= w1 <= y2 <= w2 and abs(x1-v1)<= d and abs(x2-v2)<= d:
            ERG.append(rLN_HL[ii]) 

   return(ERG)

############################################################################ 

def makeTSNew(kindOfBox, noc, xmm, m, ML, ergL, ML_bbm, ergL_bbm, correction_H, correction_V):

   for col in range(noc):
      a,b                     = xmm[col][0], xmm[col][1]     
      M_H, M_V                = valuationMatrix(ML, ergL, ML_bbm, ergL_bbm, col, kindOfBox, correction_H, correction_V)  
      
      WLH, WLV                = getattr(L_H, 'WL'+str(col)).WLH, getattr(L_V, 'WL'+str(col)).WLV    
   
############################################################################ 

def makeTS(draw, SS, l, dir, mm=2, c=250, xmax=570, ymax=820):

   #img  = Image.fromarray(D).convert('L')
   #draw = ImageDraw.Draw(img)

   l = tp(l).tolist()[0]
   for ii in range(len(SS)):
      a,b = SS[ii]
      ts  = (a+b)/2
      p   = int((1-l[ii])*c)
      if dir=='H':
         x,y = min(570, xmax), int(max(2, ts))
      else:
         x,y = int(max(2, ts)), ymax 
    
      draw.rectangle( (x-2, y-2, x+2, y+2), fill=( p,p,p ) )
      if ii%mm==0 and mm<100:         
         if dir=='V':
            draw.text( (x-4,y+10), str(ii) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
         if dir=='H':
            draw.text( (x-15,y-4), str(ii) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
            draw.text( (x+5,y-4), str(np.round(l[ii],2)) ,(100), font=ImageFont.truetype('Roboto-Bold.ttf', size=9))
            
   
   return(draw)     

############################################################################ 

def scalePlots(draw,  noc, xmm, INFO, m, kindOfBox='TA', mm=2):

   kindOfBoxAlt = kindOfBox
   if INFO.copyHL:
      kindOfBox = 'TA'
 
   OH  = getattr( getattr( getattr(INFO,  kindOfBox), 'bB'), 'H')
   OV  = getattr( getattr( getattr(INFO,  kindOfBox), 'bB'), 'V')
   #st  = OH.stepSize

   WLH = OH.WL0.WL
   WLV = OV.WL0.WL
   SSH = list(map(lambda x: x[1], WLH))
      
   kindOfBox = kindOfBoxAlt

   for ii in range(noc):
      a,b  = xmm[ii]
      l    = int(0.5*(m- (b-a)))
      mh   = getattr( getattr(INFO, kindOfBox), 'MH'+ str(ii))
      mv   = getattr( getattr(INFO, kindOfBox), 'MV'+ str(ii))[ int( l/OV.stepSize): int((l + (b-a))/OV.stepSize ) , :]
      SSV = list( filter( lambda x: a <= x[0] <=  x[1] <= b, list(map(lambda x: x[1], WLV))))
      xmax = min( 590, b)
      
      draw    = makeTS(draw, SSH, mh[:, 2], dir='H',mm=mm, c=250, xmax=b)    
      draw    = makeTS(draw, SSV, mv[:, 2], dir='V',mm=mm, c=250, xmax=b)   

   return(draw)

############################################################################ 

def makeINFO():
   print("creating INFO ...")
   INFO                      = boxMaster() 
   INFO.TA                   = boxMaster()
   INFO.TA.bB                = boxMaster()
   INFO.TA.bB.H              = boxMaster()
   INFO.TA.bB.V              = boxMaster()
   INFO.TA.bBHV              = boxMaster()
   INFO.TA.bBHV.H            = boxMaster()
   INFO.TA.bBHV.V            = boxMaster()
   INFO.HL                   = boxMaster()
   INFO.HL.bB                = boxMaster()
   INFO.HL.bB.H              = boxMaster()
   INFO.HL.bB.V              = boxMaster()
   INFO.HL.bBHV              = boxMaster()
   INFO.HL.bBHV.H            = boxMaster()
   INFO.HL.bBHV.V            = boxMaster()

   return(INFO)

###############################################################################

def putTogether(WL, withMarks=False, size=tuple([842, 596])):

   M = np.array( 255*np.ones( size), dtype='uint8')
   for ii in range(len(WL)):
      y1, y2       = WL[ii][1]
      M[ y1:y2, :] = WL[ii][0]
      if withMarks:
         M[ y1-1:y1+1, 0:30] = 0
         M[ y2-1:y2+1, 500:] = 0 
   
   return(M)

###############################################################################

def drawErg(C, WL, erg, direction ='H'):
      
   for ii in range(len(WL)):
      wl = WL[ii]
      a,b = wl[1][0], wl[1][1]

      if direction == 'H':
         C[a, :] = 0
         C[b-1:b+1, :] = 0
      
      if direction == 'V':
         C[:, a] = 0   
         C[:, b-1:b+1] = 0
               
   return(C)    

###########################################################################    


#***
#*** MAIN PART
#***
#
#  exec(open("tableFinder.py").read())
#

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train_hochkant/word/train_hochkant-21-portrait-word.png')
Ct                   = MAT.downSampling(C1, 3)                
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
#SWO_2D.outer         = False
#SWO_2D.allCoef       = True
#SWO_2D.upScaling     = False
SWO_2D.onlyCoef      = True
SWO_2D.allLevels     = False

white                = 255
black                = 0 

engine               = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
con                  = engine.connect()

# **********************
# *** load data      ***
# **********************

try:
   a = len(INFO.TA.bB.H.WL0.WL)
except:
   print("loading data...")
   INFO              = makeINFO()
   INFO.path         = '/home/markus/anaconda3/python/data/'
   INFO.kindOfImages = 'JPG'  
   INFO.white        = 255
   INFO.black        = 0    
   INFO.copyHL       = True
   INFO.flatten      = True
   date              = '25.01.2022-'
   FL                = ['TA-bB-HpT-JPG-' , 'TA-bB-VpT-JPG-' , 'TA-bBHV-HpT-JPG-','TA-bBHV-VpT-JPG-' , 'HL-bB-HpT-JPG-'    , 'HL-bB-VpT-JPG-'   , 'HL-bBHV-HpT-JPG-' ,'HL-bBHV-VpT-JPG-'   ]
   DL                = [date+'21:27:44'  , date+'21:46:53'  , date+ '22:18:47'  , date+ '22:43:32'  ,  date + '23:01:29'  , date + '23:20:09'  , date + '23:51:34'  , '26.01.2022-' + '00:16:21' ]
   
   INFO.KBOX         = ['TA','HL']
   INFO.MBOX         = ['bB', 'bBHV']
   INFO.DBOX         = ['H', 'V']

   for kindOfBox in INFO.KBOX:
      for method in INFO.MBOX:
         for direction in INFO.DBOX:
             
            fname = kindOfBox + '-' + method + '-' + direction + 'pT-' + INFO.kindOfImages + '-'
            try:
               date  = DL[ FL.index(fname)]
            except:
               print("Cant find date for file " + fname + "!")
            
            DATA = MISC.loadIt(INFO.path + fname + date)
            O    = getattr( getattr(INFO,  kindOfBox), method)
            setattr(O, direction,  getattr( getattr( getattr(DATA.INFO,  kindOfBox), method), direction)) 
            S    = getattr(O, direction)
            setattr(S, 'rf', DATA.rf)
            try:
               INFO.onlyWhiteBlack  = DATA.INFO.onlyWhiteBlack
               INFO.wBB             = DATA.INFO.wBB
            except:
               print("INFO contains no information about onlyBlackWhite option...seting values to default!")
               INFO.onlyWhiteBlack = False
               INFO.wBB            = 180

   print("...done")



### STPE

STPE                      = dOM.stripe(C1, stepSize=0, windowSize=0, direction='H', SWO_2D=SWO_2D)
STPE.dd                   = 0.20
STPE.tol                  = 30

stepSize_H                = 5
windowSize_H              = 30
INFO.TA.bB.H.stepSize     = stepSize_H 
INFO.TA.bB.H.windowSize   = windowSize_H 
INFO.TA.bBHV.H.stepSize   = stepSize_H 
INFO.TA.bBHV.H.windowSize = windowSize_H 
 
windowSize_V              = 80
stepSize_V                = 5
INFO.TA.bB.V.windowSize   = windowSize_V 
INFO.TA.bB.V.stepSize     = stepSize_V 
INFO.TA.bBHV.V.windowSize = windowSize_V  
INFO.TA.bBHV.V.stepSize   = stepSize_V  


setattr(INFO.TA, 'correction-H', 0.45)  #0.35
setattr(INFO.TA, 'correction-V', 0.3)
setattr(INFO.TA, 'weightbBHV-V', 0.5)
setattr(INFO.TA, 'weightbB-V'  , 0.5)
setattr(INFO.TA, 'weightbBHV-H', 0.5)
setattr(INFO.TA, 'weightbB-H'  , 0.5)

setattr(INFO.HL, 'correction-H', 0.1)
setattr(INFO.HL, 'correction-V', 0.2)
setattr(INFO.HL, 'weightbBHV-V', 0.5)
setattr(INFO.HL, 'weightbB-V'  , 0.5)
setattr(INFO.HL, 'weightbBHV-H', 0.5)
setattr(INFO.HL, 'weightbB-H'  , 0.5)



ss                        = input("calculate SWCs (Y/N) ?")
calcSWCs                  = False
if ss=='Y':
   calcSWCs = True
#calcSWCs                  = False
calculateJPGs             = True
withScalePlot             = True

np.set_printoptions(suppress=True)


dpi              = 200
challenge        = dOM.JPGNPNGGenerator('/home/markus/anaconda3/python/pngs/challenge/',  'challenge' , '/home/markus/anaconda3/python/pngs/challenge/word/' , 'challenge' , 1, 0, False, dpi, 'cv')   
challenge.Q      = []
challenge.engine = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
challenge.con    = engine.connect() 
train            = dOM.JPGNPNGGenerator('/home/markus/anaconda3/python/pngs/train/', 'train', '/home/markus/anaconda3/python/pngs/train/word/', 'train', 1, 0, False, dpi, 'cv')  
columns          = dOM.columns()
P                = [[1,0]]  
#P                = [[267,1]]


source           = challenge

for dd in 1*P:
   page, noc           = dd
   ## noc ist hier nur für den Fall, dass man eine Anzahl von Spalten erzwingen möchte
   fname               = source.outputFolder + source.outputFile + '-' + str(page) + '-portrait-word' 
   fname_bbm           = fname + '-bbm'
   Corg                = np.matrix( MAT.generateMatrixFromImage(fname+'.jpg'), dtype='uint8')
   noc, _,_,_,_        = columns.coltrane2(Corg)

   DATA.n, DATA.m      = Corg.shape
   DATA.CL, DATA.xmm   = STPE.genMat(fname, noc, 'bB', challenge, INFO.onlyWhiteBlack, INFO.wBB)  
   DATA.CL_bbm, _      = STPE.genMat(fname, noc, 'bBHV', challenge, INFO.onlyWhiteBlack, INFO.wBB)
  
   if calcSWCs:
      INFO                = decomposeMatrices(DATA, STPE, INFO)
      INFO                = calculateSWCs(DATA, STPE, INFO, des='')
      INFO                = applyRF(DATA, INFO, des='')
   
   rLN_TA              = allBoxes('TA', noc, INFO, DATA.xmm, DATA.m, 2)
   rLN_HLt             = allBoxes('HL',noc, INFO, DATA.xmm, DATA.m)
   rLN_HL              = filterHL(rLN_TA, rLN_HLt, 30)
  
   img_TA, draw_TA     = makeImage(Corg)
   img_HL, draw_HL     = makeImage(Corg)

   if withScalePlot:
      draw_TA = scalePlots(draw_TA, noc, DATA.xmm, INFO, DATA.m, kindOfBox='TA', mm=5 )
      draw_HL = scalePlots(draw_HL, noc, DATA.xmm, INFO, DATA.m, kindOfBox='HL', mm=5 )

   for ii in range(len(rLN_TA)):
      r = rLN_TA[ii][0]       
      draw_TA.rectangle(r, outline ="red",width=3)
     
   for ii in range(len(rLN_HL)):
      r = rLN_HL[ii][0]      
      draw_TA.rectangle(r, outline ="blue",width=3)    

   for ii in range(len(rLN_HLt)):
      r = rLN_HLt[ii][0]      
      draw_HL.rectangle(r, outline ="blue",width=3) 
    
   #C, ERG = allColumnsOfTables(C2=np.array(Corg), img=img, rL=rLN_TA, GEN=source, page=page, xmm=DATA.xmm, white=255)

   taWeights           = str( getattr(INFO.TA, 'weightbB-H'))      + '/'        + str( getattr( INFO.TA, 'weightbBHV-H')) + ' - ' + str(getattr( INFO.TA, 'weightbB-V')) + '/' + str(getattr(INFO.TA, 'weightbBHV-V'))
   taCorr              = str( getattr(INFO.TA, 'correction-H'))    + "/"        + str( getattr(INFO.TA, 'correction-V'))
   hlWeights           = str( getattr(INFO.HL, 'weightbB-H'))      + '/'        + str( getattr( INFO.HL, 'weightbBHV-H')) + ' - ' + str(getattr( INFO.HL, 'weightbB-V')) + '/' + str(getattr(INFO.HL, 'weightbBHV-V'))
   hlCorr              = str( getattr(INFO.HL, 'correction-H'))    + "/"        + str( getattr(INFO.HL, 'correction-V'))
   ss                  = "page:" + str(page) + "  noc:" + str(noc) + "  TA-W: " + taWeights + "  Corr:" + taCorr # + "  HL-W:" + hlWeights + "  Corr:" + hlCorr 
   draw_TA.text( (20,0), ss, (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))
   ss                  = "page:" + str(page) + "  noc:" + str(noc) + "  HL-W:" + hlWeights + "  Corr:" + hlCorr 
   draw_HL.text( (20,0), ss, (255,0,255),font=ImageFont.truetype('Roboto-Bold.ttf', size=12))
   #Image.fromarray(C).show() 
   #C = scalePlots(DATA.CL[1][0], 1, [[0, 596]], INFO);  Image.fromarray(C).show()

   img_TA.show()
   #img_HL.show()

#WL             = INFO.TA.bB.V.WL0.WL
#C              = drawErg(np.array(img), WL , INFO.TA.bB.V.erg0, 'V')
