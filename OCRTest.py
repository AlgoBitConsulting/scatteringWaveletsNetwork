import sys, subprocess
from os import system
sys.path.append('/home/markus/anaconda3/python/development/modules')
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont, ImageFilter
import timeit

from datetime import datetime 
import pdb
from sys import argv
from copy import deepcopy
from tqdm import tqdm

import misc_v9 as MISC
import dataOrganisationModule_v3 as dOM
import morletModule_v2 as MM  
import DFTForSCN_v7 as DFT
import scatteringTransformationModule_2D_v9 as ST
import predictionModule_v2 as PM

from sklearn.ensemble import RandomForestClassifier
import cv2 as cv

import pandas as pd

from sklearn import metrics

############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

############################################################################# 

def enlargeGreyMatrix( M, n, m):
   C      = makeWhiteUint8( n,m )
   nt, mt = M.shape
   if nt <= n and mt <= m:
      dn, dm = int( 0.5*(n-nt)), int(0.5*(m-mt))
      C[dn:dn+nt, dm:dm+mt] = M

   return(C)

#############################################################################

def makeWhiteUint8( n,m ):
   R = np.array( 255*np.ones( (n,m)), dtype='uint8')
   return(R)

#############################################################################

def WLtoERG(nn, AL, al):   # nn= len(WL)
 
   CL = []
   for ii in range(nn):
      a = [ AL[ii], al[ii], ii]
      CL.append(a)   

   ERG = [CL, 'test']

   return(ERG)

############################################################################

def plotCharacters(char, ANNO):

   erg  = []
   for ii in range(len(ANNO)):
      ann = ANNO[ii]
      if ann == char:
         erg.append([MATL[ii], ann])   
         Image.fromarray(MATL[ii]).show() 

############################################################################

def plotCharactersDetailed(ML, AL, char):

   erg  = []
   l    = []
   imgL = []
   for ii in range(len(AL)):
      if AL[ii][0] == char:
         l.append(ii)

   for ii in l:
      M, j, i, k = ML[ii]
      A, j, i, k = AL[ii]
      #print(ii)
      M  = cv.threshold(M, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
      C = enlargeGreyMatrix(M, 100,100)
      erg.append([ C, A, j, i, k])      
      img  = Image.fromarray(C)
      draw = ImageDraw.Draw(img)
      draw.text( (2,2), str(j) + ':' + str(i) + ':' + str(k), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=9))
      imgL.append(img)
      
   return([imgL, erg,l])

############################################################################

def makeOverview(ANNO, dstr1,n ):

   OCR          =  PM.TD("bB")
   OCR.loadData('/home/markus/anaconda3/python/data/', 'testERG-' + dstr1)
   OCR.evaluate(n=n, write=False, crossVal=False, dict=True, pri=False)

   lenSignsInRF = len(list(set(OCR.l2)))
   annoCount    = []
   annoSigns    = list(set(ANNO))
   for ii in range(len(annoSigns)):
      annoCount.append([annoSigns[ii], ANNO.count(annoSigns[ii])])   
   annoCount.sort(key=lambda x: x[1])

   DICT = OCR.erg 
   K    = list(DICT.keys())
   K.remove('accuracy')
   K.remove('macro avg')
   K.remove('weighted avg')

   M = []
   for ii in range(len(K)):
      char    = K[ii]
      total   = list(filter( lambda x: x[0] == char, annoCount))[0][1]
      ss      = DICT[char]
      supp    = ss['support']
      sIP     = np.round(supp/total,2)
      prec    = np.round(ss['precision'],2)
      rec     = np.round(ss['recall'],2)
      f1score = np.round(ss['f1-score'],2)

      M.append([ char, total, sIP,  '|', prec, rec, f1score, supp] )

   mA   = DICT['macro avg']
   wA   = DICT['weighted avg']

   M.append(['---', '---', '---', '|', '---', '---', '---', '---'])
   M.append(['accuracy', len(annoCount),    np.round(lenSignsInRF/len(annoCount),2),  '|', DICT['accuracy'] , '', '', ''] )
   M.append(['macro avg', len(annoCount),   np.round(lenSignsInRF/len(annoCount),2),  '|', np.round(mA['precision'],2),  np.round(mA['recall'],2), np.round(mA['f1-score'],2), mA['support'] ])
   M.append(['weighed avg', len(annoCount), np.round(lenSignsInRF/len(annoCount),2),  '|', np.round(wA['precision'],2),  np.round(wA['recall'],2), np.round(wA['f1-score'],2),wA['support'] ])

   A = pd.DataFrame( M, columns=['label', 'total', 'support/total', '|', 'precision', 'recall', 'f1-score', 'support'])   
   return([A, annoSigns, annoCount, DICT])

############################################################################

def makeLists(L, excp):

   CLt= []
   for l in L:
      A = np.concatenate(l)
      CLt.extend(A)

   CL = []
   for clt in CLt:
      if clt[1] not in excp:
         CL.append(clt)

   MATL = []
   ANNO = []
   for ii in range(len(CL)):
      M = CL[ii][0]
      if len(M.shape)>2:
         M = M[:,:,0]   

      M  = np.array( 255*np.array( M > 0, dtype='int'), dtype='uint8')

      if not(imageIsEmpty(M)):
         anno = CL[ii][1]
         ANNO.append(anno)
         C = enlargeGreyMatrix(M, 100,100)
         MATL.append(C)

      
   return([MATL, ANNO])

############################################################################

def makeListsDetailed(BIG, excp):

   ML = []
   AL = []
   for j in range(len(BIG)):
      L = BIG[j]
      for i in range(len(L)):
         w = L[i]
         for k in range(len(w)):
            M = w[k][0]
            if len(M.shape) >2:
               M = M[:,:,0]
            
            M  = np.array( 255*np.array( M > 0, dtype='int'), dtype='uint8')
            if not(imageIsEmpty(M)) and w[k][1] not in excp:
               C = enlargeGreyMatrix(M, 100,100)
               ML.append( [C, j,i,k]) 
               AL.append( [w[k][1], j,i,k])

   return([ML, AL])

############################################################################

def showAllImages(IL):
   for img in IL:
      img.show()

############################################################################

def imageIsEmpty(N, white=255):

   imageIsEmpty = False

   r = list(set(np.concatenate(N.tolist())))
   if len(r)==1 and r[0] == white:
      imageIsEmpty = True

   return(imageIsEmpty)   

############################################################################

def predictChars(rf, MLt, ALt, char):

   imgL, erg, l = plotCharactersDetailed(MLt, ALt, char)
   a          = list(map( lambda x: x[0], erg))
   WLt        = MISC.makeIt(a, SWO_2D, "rf") 
   predicted  = rf.predict(WLt)
   P          = rf.predict_proba(WLt)

   l2 = []
   for ii in range(len(predicted)):
      if predicted[ii] != char:
         l2.append(ii)

   return([predicted, P, l2])

############################################################################

def makeLabel():
 
   global SWO_2D

   ss  = "-SWC"  + "-initEta="+str(SWO_2D.init_eta) + "-J=" + str(SWO_2D.J) +"-nang=" + str(SWO_2D.nang) + "-ll=" + str(SWO_2D.ll) + "-m=" + str(SWO_2D.m) + "-normalization=" + str(SWO_2D.normalization)

   return(ss)

############################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("OCRTest.py").read())
#

adaptMatrixCoef      = tuple([100, 100])
downSamplingRate     = 0

MATL1                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-1-16.02.2022-15:32:02')
MATL2                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-2-16.02.2022-16:29:11')
MATL3                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-3-16.02.2022-17:10:48')
MATL4                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-4-16.02.2022-17:49:37')
MATL5                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-5-16.02.2022-11:21:06')
MATL6                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-6-16.02.2022-21:32:47')
MATL7                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-7-16.02.2022-22:06:30')
MATL8                 = MISC.loadIt('/home/markus/anaconda3/python/data/MATL_total-8-16.02.2022-23:03:27')

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = enlargeGreyMatrix(MATL1[0][0][0],100,100)
train                = dOM.JPGNPNGGenerator('/home/markus/anaconda3/python/pngs/train/', 'train', '/home/markus/anaconda3/python/pngs/train/word/', 'train', 1, 0, False, 500, 'cv')     
Ct                   = MAT.downSampling(C1, downSamplingRate)                
dx,dy                = 0.15, 0.15
SWO_2D               = MM.SWO_2D(Ct, round(Ct.shape[1]*0.5*dx,3), round(Ct.shape[0]*0.5*dy,3))
SWO_2D.init_eta      = 2.5
SWO_2D.kk            = 1
SWO_2D.sigma         = mat([[SWO_2D.kk,0],[0,SWO_2D.kk]])
SWO_2D.J             = 2
SWO_2D.nang          = 16
SWO_2D.ll            = 3
SWO_2D.jmax          = SWO_2D.J
SWO_2D.m             = 2
SWO_2D.normalization = False    # (m=2 wird mit m=1-Wert normalisiert)
SWO_2D.onlyCoef      = True  # für debugging Zwecke auf False setzen
SWO_2D.allLevels     = False   # wenn True dann werden nur Werte für m=2, ansonsten auf m=1 geliefert (DS enthält trotzdem alles für debugging Zwecke)

#ss                   = "SWC"  + "-initEta="+str(SWO_2D.init_eta) + "-J=" + str(SWO_2D.J) +"-nang=" + str(SWO_2D.nang) + "-ll=" + str(SWO_2D.ll) + "-m=" + str(SWO_2D.m) + "-normalization=" + str(SWO_2D.normalization)

np.set_printoptions(suppress=True)

L           = [MATL1, MATL2, MATL3, MATL4, MATL5, MATL6, MATL7, MATL8]
excp        = ['/', 'x', 'en', 'C', ')', 'F', 'nd', 'I', 'ng', 'am', 'y', '00', 'Ü', 'se', '%', 'P', '(', ':', 'O', 'T', '€', 'W', 'L']
#excp       = ['/', 'en', ')', 'nd', 'ng', 'am', '00', 'se', '%', '(', ':', '€']
MLt2, ALt2 = makeListsDetailed(L, excp)

MATLt1      = list(map(lambda x: x[0], MLt2))
ALt1        = list(map(lambda x: x[0], ALt2))
alt1        = []
alt1        = ALt1


## ii = 0-7: resize(100,70)
## ii = 8: resize(70,49)
## ii = 9: resize(70,49)

## ii = 10: N           = np.array(Image.fromarray(cut(M)).resize((70,80)))
try: 
   a = len(OCR.L1)
except:
   OCR          =  PM.TD("bB")   
   #OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=3-nang=16-ll=3-m=2-normalization=False-18.03.2022-08:26:12')
   #OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=3-nang=16-ll=3-m=2-normalization=False-22.03.2022-21:44:40')
   #OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=2-nang=16-ll=3-m=2-normalization=False-23.03.2022-14:33:31')
   #OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=2-nang=8-ll=2-m=2-normalization=False-25.03.2022-11:25:01')
   OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=2-nang=16-ll=3-m=2-normalization=False-25.03.2022-13:58:20')
   OCR.evaluate(n=2000, write=False, crossVal=False, dict=False, pri=False)
   t            = list(OCR.rf.classes_)



def cut(Nt, size = (50,50), enlargeSize = (100,100), white=255, resizeByRatio=False):

   N      = Nt.copy()
   N      = cv.threshold(N, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   cols   = N.sum(axis=0)
   lc     = np.where( cols < white*N.shape[0])[0]   
   startc = max( 0, lc[0]-1)
   endec  = min( N.shape[1]-1, lc[len(lc)-1]+1)

   rows   = N.sum(axis=1)
   lr     = np.where( rows < white*N.shape[1])[0]   
   startr = max( 0, lr[0]-1)
   ender  = min( N.shape[0]-1, lr[len(lr)-1]+1)

   A      = N[startr:ender, startc:endec ] 

   if resizeByRatio:
      r      = (ender-startr)/(endec-startc)   

      if (ender-startr) >= (endec-startc):
         r = (endec-startc)/(ender-startr)
         size = (int(r*size[0]), size[1])
      else:
         size = (size[0], int(r*size[1]))

   B = Image.fromarray(A).resize(size)
   C = enlargeGreyMatrix( np.array(B), enlargeSize[0], enlargeSize[1])

   return([A, C])


# 'a' ii = 36    [50,50]
# 'b' ii = 16    [45,50]
# 'c' ii = 742   [50,50] schwer
# 'd' ii = 26    [45,50]
# 'e' ii = 34    [50,55]
# 'f' ii = 64    [40,40]
# 'g' ii = 33    [50,45]
# 'h' ii = 12    [50,50]
# 'i' ii = 214   [50,40]
# 'j' problem
# 'k' ii = 116   [45,50]
# 'l' ii = 735   [40,45]
# 'm' ii = 7     [45,55]
# 'n' ii = 18    [45,50]
# 'o' ii = 718   [50,50]
# 'p' ii = 1994  [45,50]
# 'q' ii =       kein 'q' in Liste 
# 'r' ii = 1122  [45,45] schwer
# 's' ii = 37    [50,50]
# 't' ii = 48    [45,45]
# 'u' ii = 63    [40,55]
# 'v' ii = 134   [45,50] 
# 'w' ii =       [45,55]
# 'x' ii =       kein 'x' in Liste
# 'y' ii =       kein 'yy' in Liste
# 'z' ii = 287   [55, 40]



 
 
 
def modifyCharacter(character):
#ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'Ä', 'Ö','Ü' ]
#alphabet = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'r', 't', 'y', 'ä']
   erg = character

   if character == 'o':
      erg = 'O'   
   if character == 's':
      erg = 'S'
   if character == 'c':
      erg = 'C'
   if character == 'p' or character == 'P':
      erg = 'd'
   if character == 'u' or character=='n':
      erg = 'U'
   if character == 'v':
      erg = 'V'
   if character == 'w' or character=='W':
      erg = 'M'
   if character == 'x':
      erg = 'X'
   if character == 'z':
      erg = 'Z'
   if character=='q':
      erg = 'b'

   return(erg)



ii          = 1594
M           = MATLt1[ii].copy()
BL          = []
T           = list(OCR.rf.classes_)
character   = modifyCharacter( alt1[ii])
nn          = T.index(character)

for xx in range(40, 60, 5):
   for yy in range(40, 60, 5):
      size        = (xx,yy)
      A,C         = cut(M, size=size)
      BL.append([A,C,xx,yy])

CL   = list(map(lambda x: x[1], BL))
WL   = MISC.makeIt(CL, SWO_2D, "rf") 
r    = OCR.rf.predict_proba(WL)
r    = list(r)
s    = list(OCR.rf.predict(WL))
M    = np.matrix(r)
maxL = []

for jj in range(M.shape[1]):
   maxL.append( max(M[:, jj]).tolist()[0][0])

t1 = list(np.where(np.array(maxL)==max(maxL))[0])
Z = list(map(lambda x: T[x], t1)) #T[ maxL.index(max(maxL))]
print("Z='" + str(Z) + "'")


l = maxL.copy()
l.sort(reverse=True)

t = list(set(list(filter( lambda x: x/l[0]>=0.7, l))))

chr = []
for x in t:
   t1 = list(np.where(np.array(maxL)==x)[0])
   erg = list(map(lambda x: [ T[x], s.count(T[x]), maxL[x]], t1))
   chr.extend(erg)

print(chr)


ERG = []
for jj in range(len(s)):
   if s[jj] == character:
      ERG.append([BL[jj][2], BL[jj][3], r[jj][nn]])

"""

#A,C         = cut(MATLt1[ii], size=(50, 50))
#WL          = MISC.makeIt([C], SWO_2D, "rf") 
#r           = OCR.rf.predict_proba(WL)
#r           = list(r)
#s           = list(OCR.rf.predict(WL))
   
"""

p = np.matrix(r).mean(axis=0).tolist()[0]



#A    = interpolate.interp2d(SWO_2D.X, SWO_2D.Y, real(C_DFFT)*F_r, kind='cubic')
