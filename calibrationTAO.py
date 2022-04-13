import sys, subprocess
from os import system
sys.path.append('/home/markus/anaconda3/python/development/modules')
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import timeit
import misc_v9 as MISC
import dataOrganisationModule_v3 as dOM
import morletModule_v2 as MM  
from datetime import datetime 
import pdb
from sys import argv
from copy import deepcopy
from tqdm import tqdm
import mysql.connector
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
from sklearn.ensemble import RandomForestClassifier

############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

############################################################################# 

class boxMaster:
   def __init__(self, name=''):
      self.name            = name  

###########################################################################

def makeboxMaster():    

   INFO                        = boxMaster() 
   INFO.kindOfImages           = 'JPG'                       
   INFO.TA                     = boxMaster()
   INFO.TA.bB                  = boxMaster()
   INFO.TA.bB.H                = boxMaster()
   INFO.TA.bB.V                = boxMaster()
   INFO.TA.bBHV                = boxMaster()
   INFO.TA.bBHV.H              = boxMaster()
   INFO.TA.bBHV.V              = boxMaster()
   INFO.HL                     = boxMaster()
   INFO.HL.bB                  = boxMaster()
   INFO.HL.bB.H                = boxMaster()
   INFO.HL.bB.V                = boxMaster()
   INFO.HL.bBHV                = boxMaster()
   INFO.HL.bBHV.H              = boxMaster()
   INFO.HL.bBHV.V              = boxMaster()
   
   return(INFO)  


###########################################################################

def makeA(WL, wBB=180):

   H   = []
   R   = tqdm(range(len(WL)))
   for ii in R:
      M = np.array( 255*np.array( WL[ii][0] >= wBB, dtype='int'), dtype='uint8')
      h = np.histogram(M, bins=list(range(0,257)))
      H.append([h[0], ii])
 
   A = list(map(lambda x: x[0], H))
   return(A)
   
###########################################################################

def makeUnique(WL, usePercentageTreshold=False, pT=0.05, wBB=180):
  
   A = makeA(WL, wBB)

   if usePercentageTreshold:
      print("makeUnique: using precentagTreshold...")
      a1 = list(map( lambda x: [round( A[x][0]/A[x][255], 2), x], list(range(len(A))) ))
      b1 = list(filter( lambda x: x[0] >= pT, a1))
      B_ind = list(map( lambda x: x[1], b1))
   else:
      print("makeUnique: using histogram...")
      B_ind, B_rev, B_inv, B_cou = np.unique(A, return_index=True, return_inverse=True, return_counts=True, axis=0)

   WLt = []
   for ii in range(len(B_ind)):
      WLt.append(WL[B_ind[ii]])

   return(WLt)
   
################################################################################################


def makeAnnotations(STPE, INFO, L, COLN, con, des='annotations...', overRideColumnsDetection=False):

   global train

   WL         = []
   R          = tqdm(1*L)
   R.set_description(des)
   
   for l in R:
      hashValue           = l[ COLN.index('hashValueJPGFile')]  
      noc                 = l[ COLN.index('numberOfColumns')]
      fname               = l[ COLN.index('filenameJPG')].split('.')[0]
      page                = l[ COLN.index('page')]

      if INFO.overRideColumnsDetection:
         noc = 1

      CL, xmm             = STPE.genMat(fname, noc, INFO.method, train, INFO.onlyWhiteBlack, INFO.wBB)
      K                   = STPE.generateLabelBoxes(INFO.kindOfBox, hashValue, con) 

      #if INFO.kindOfBox =='HL':
      #   K_HL           = K.copy()
      #   INFO.kindOfBox = 'TA'
      #   K_TA           = STPE.generateLabelBoxes(INFO.kindOfBox, hashValue, con) 
      #   INFO.kindOfBox = 'HL'
          
      #   K = K_HL + K_TA

      w, _, _             = STPE.makeWL(CL, K, xmm, page, hashValue)
      WL.extend(w)   

   return(WL)

################################################################################################

def makeRF(STPE,WL, INFO, des = ''):

   print(des)
   AL, al          = STPE.prepareData(WL, des)
   rf              = RandomForestClassifier(n_estimators=1000)
   rf.fit(AL, al)
   print("...done")
       
   return([rf, AL, al])

################################################################################################

def makeTestData(L, COLN, INFO, STPE_RF_H, STPE_RF_V, con):

   for l in L:
      hashValue = l[ COLN.index('hashValueJPGFile')]  
      noc       = l[ COLN.index('numberOfColumns')]
      fname     = l[ COLN.index('filenameJPG')].split('.')[0]
      page      = l[ COLN.index('page')]
 
      CL, xmm        = STPE_RF_H.genMat(fname, noc, INFO, train)  
      INFO.kindOfBox = 'TA' 
      K_TA           = STPE_RF_H.generateLabelBoxes(INFO, hashValue, con)
      INFO.kindOfBox = 'HL'
      K_HL           = STPE.generateLabelBoxes(INFO, hashValue, con) 
      K = K_TA + K_HL
      
      STPE_RF_H.windowSize = INFO.TA.bB.H.windowSize
      STPE_RF_H.stepSize   = INFO.TA.bB.H.stepSize
      WLH, IMGL_H, ERGL_H = STPE_RF_H.makeWL(CL, K, xmm, page, hashValue)

      STPE_RF_V.windowSize = INFO.TA.bB.V.windowSize
      STPE_RF_V.stepSize   = INFO.TA.bB.V.stepSize
      WLV, IMGL_V, ERGL_V = STPE_RF_V.makeWL(CL, K, xmm, page, hashValue)
    
      return([CL, IMGL_H, ERGL_H, ERGL_V, K])

################################################################################################

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

def gEOM(M):
   erg = list(set(np.concatenate(M.tolist())))
   return(erg)

###############################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("calibrationTAO.py").read())
#

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train_hochkant/word/train_hochkant-21-portrait-word.png')
train                = dOM.JPGGenerator('/home/markus/anaconda3/python/pngs/train/', 'train', '/home/markus/anaconda3/python/pngs/train/word/', 'train', 1, 0, False, 500, 'cv')     
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
SWO_2D.outer         = False
SWO_2D.allCoef       = True
SWO_2D.upScaling     = False
SWO_2D.onlyCoef      = True
SWO_2D.allLevels     = False

engine               = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
con                  = engine.connect()

###########################################################################

SQL = "(select * from TAO where namePDFDocument != '/home/markus/anaconda3/python/pngs/challenge/challenge' and format='portrait' and what='word' and original='YES' and (T1 is not null and length(replace(T1, ' ',  ''))>0)) union "
SQL = SQL + "(select * from TAO where namePDFDocument != '/home/markus/anaconda3/python/pngs/challenge/challenge' and format='portrait' and what='word' and original='YES' and hasTable=0 order by page )"
#SQL = "select * from TAO where namePDFDocument = '/home/markus/anaconda3/python/pngs/train/lf-gb2019finalg-2-columns-pages-with-at-least-one-table' and format='portrait' and what='word' and original='YES' and (T1 is not null and length(replace(T1, ' ',  ''))>0) and page=4"

rs                          = con.execute(SQL)
COLN                        = list(rs.keys())
L                           = list(rs)

DATA                          = boxMaster()

INFO                          = boxMaster() 
INFO.kindOfImages             = 'JPG'       
INFO.overRideColumnsDetection = False
INFO.SQL                      = SQL
INFO.wBB                      = 180     
INFO.onlyWhiteBlack           = True
INFO.usePercentageTreshold    = True
INFO.pT                       = 0.25        

INFO.TA                     = boxMaster()
INFO.TA.bB                  = boxMaster()
INFO.TA.bB.H                = boxMaster()
INFO.TA.bB.V                = boxMaster()
INFO.TA.bBHV                = boxMaster()
INFO.TA.bBHV.H              = boxMaster()
INFO.TA.bBHV.V              = boxMaster()
INFO.HL                     = boxMaster()
INFO.HL.bB                  = boxMaster()
INFO.HL.bB.H                = boxMaster()
INFO.HL.bB.V                = boxMaster()
INFO.HL.bBHV                = boxMaster()
INFO.HL.bBHV.H              = boxMaster()
INFO.HL.bBHV.V              = boxMaster()




INFO.TA.bB.H.stepSize           = 5
INFO.TA.bB.H.windowSize         = 30
INFO.TA.bB.H.adaptMatrixCoef    = tuple([106, 76])
INFO.TA.bB.H.downSamplingRate   = 3
INFO.TA.bB.H.pT                 = 0.2

INFO.TA.bB.V.stepSize           = 5
INFO.TA.bB.V.windowSize         = 80
INFO.TA.bB.V.adaptMatrixCoef    = tuple([106, 76])
INFO.TA.bB.V.downSamplingRate   = 3
INFO.TA.bB.V.pT                 = 0.1

INFO.TA.bBHV.H.stepSize         = 5
INFO.TA.bBHV.H.windowSize       = 30
INFO.TA.bBHV.H.adaptMatrixCoef  = tuple([106, 76])
INFO.TA.bBHV.H.downSamplingRate = 3
INFO.TA.bBHV.H.pT               = 0.01

INFO.TA.bBHV.V.stepSize         = 5
INFO.TA.bBHV.V.windowSize       = 80
INFO.TA.bBHV.V.adaptMatrixCoef  = tuple([106, 76])
INFO.TA.bBHV.V.downSamplingRate = 3
INFO.TA.bBHV.V.pT               = 0.01



INFO.HL.bB.H.stepSize           = 5
INFO.HL.bB.H.windowSize         = 30
INFO.HL.bB.H.adaptMatrixCoef    = tuple([106, 76])
INFO.HL.bB.H.downSamplingRate   = 3
INFO.HL.bB.H.pT                 = 0.2

INFO.HL.bB.V.stepSize           = 5
INFO.HL.bB.V.windowSize         = 80
INFO.HL.bB.V.adaptMatrixCoef    = tuple([106, 76])
INFO.HL.bB.V.downSamplingRate   = 3
INFO.HL.bB.V.pT                 = 0.1

INFO.HL.bBHV.H.stepSize         = 5
INFO.HL.bBHV.H.windowSize       = 30
INFO.HL.bBHV.H.adaptMatrixCoef  = tuple([106, 76])
INFO.HL.bBHV.H.downSamplingRate = 3
INFO.HL.bBHV.H.pT               = 0.01

INFO.HL.bBHV.V.stepSize         = 5
INFO.HL.bBHV.V.windowSize       = 80
INFO.HL.bBHV.V.adaptMatrixCoef  = tuple([106, 76])
INFO.HL.bBHV.V.downSamplingRate = 3
INFO.HL.bBHV.V.pT               = 0.01

STPE_RF_H                       = dOM.stripe(C1, stepSize=0, windowSize=0, direction='H', SWO_2D=SWO_2D)
STPE_RF_H.dd                    = 0.20
STPE_RF_H.tol                   = 30

STPE_RF_V                       = dOM.stripe(C1, stepSize=0, windowSize=0, direction='V', SWO_2D=SWO_2D)
STPE_RF_V.dd                    = 0.20
STPE_RF_V.tol                   = 30




#L                           = L[0:2]
test                        = False
makeData                    = not(test)



if makeData:
   KBOX                        = ['TA', 'HL']
   MBOX                        = ['bB', 'bBHV']
   DBOX                        = ['H', 'V']
   OL                          = [STPE_RF_H, STPE_RF_V]

   ask = input("answer questions (Y/N)?")

   if ask =='Y': 
      ihl                         = input("TA/HL/BO(th) (TA/HL/BO) ?")
      if ihl == 'TA':
         KBOX.remove('HL')
      if ihl == 'HL':
         KBOX.remove('TA')
   
      ibb                         = input("bB, bBHV or both (bB/bBHV/both) ?")   
      if ibb == 'bB':
         MBOX.remove('bBHV')
      if ibb == 'bBHV':
         MBOX.remove('bB')

      dir                         = input("H, V, both (H/V/B) ?")
      if dir == 'H':
         DBOX.remove('V')
         OL.remove(STPE_RF_V)
      if dir == 'V':
         DBOX.remove('H')   
         OL.remove(STPE_RF_H)
     
   saveWL                      = input("Save also data (Y/N)?")
   nameFileAdd                 = input("additional filename:")

   for kindOfBox in KBOX:
      for method in MBOX:
         for ol in OL:
            INFO.kindOfBox      = kindOfBox 
            INFO.method         = method
            INFO.directions     = ol.direction
            O                   = getattr( getattr( getattr(INFO,  INFO.kindOfBox), INFO.method), INFO.directions)
            ol.stepSize         = O.stepSize
            ol.windowSize       = O.windowSize
            ol.adaptMatrixCoef  = O.adaptMatrixCoef
            ol.downSamplingRate = O.downSamplingRate  
            ss                  = 'annotations for '     + INFO.kindOfBox + '-' + INFO.method + '-' + INFO.directions
            tt                  = 'calculation SWC for ' + INFO.kindOfBox + '-' + INFO.method + '-' + INFO.directions
            WLt                 = makeAnnotations(STPE=ol, INFO=INFO, L=L, COLN=COLN, con=con, des=ss)
            WL                  = makeUnique(WLt, INFO.usePercentageTreshold, O.pT, INFO.wBB )
            DATA.rf, AL, al     = makeRF(STPE=ol, WL=WL, INFO=INFO, des=tt)
            if saveWL == 'Y':
               DATA.WL          = WL
               DATA.WLt         = WLt
               DATA.AL, DATA.al = AL, al
               
            DATA.INFO           = INFO
            DATA.KBOX           = KBOX
            DATA.MBOX           = MBOX
            DATA.DBOX           = DBOX

            dstr                = MISC.saveIt(DATA, '/home/markus/anaconda3/python/data/' + INFO.kindOfBox+ '-' + INFO.method+ '-' + INFO.directions + nameFileAdd+'-JPG' )   

if test: 
   INFO.kindOfBox = 'HL'; INFO.method = 'bB'; INFO.directions = 'H'
   #CL, IMGL_H, ERGL_H, ERGL_V, K   = makeTestData([L[103]], COLN, INFO, STPE_RF_H, STPE_RF_V, con)
   
   ss                  = 'annotations for '     + INFO.kindOfBox + '-' + INFO.method + '-' + INFO.directions
   tt                  = 'calculation SWC for ' + INFO.kindOfBox + '-' + INFO.method + '-' + INFO.directions
   ol                  = STPE_RF_H
   ol.stepSize         = 5
   ol.windowSize       = 30
   ol.adaptMatrixCoef  = tuple([106, 76])
   ol.downSamplingRate = 3  
   WLt                 = makeAnnotations(STPE=ol, INFO=INFO, L=L, COLN=COLN, con=con, des=ss)
   WL                  = makeUnique(WLt, INFO.usePercentageTreshold, INFO.pT, INFO.wBB )
   #WL                  = makeUnique(WLt)
   
   #A                  = makeA(WLt)
   #B_ind, B_rev, B_inv, B_cou = np.unique(A, return_index=True, return_inverse=True, return_counts=True, axis=0)

   #D = list(filter( lambda x: np.all( A[x]== A[644]), list(range(len(A)))))




