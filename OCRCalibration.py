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

import pywt

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

def matrixToRGBImage(M):

   size = ( M.shape[1], M.shape[0])
   img  = Image.new(mode="RGB",size=size, color=(255,255,255))
   A    = np.array(img)
   A[:,:,0] = M
   A[:,:,1] = M
   A[:,:,2] = M

   img = Image.fromarray(A)

   return([img, A]) 

############################################################################

def flipImage(img, t=0):

   M1  = np.array(img)
   if len(M1.shape)==3:
      M1 = M1[:,:,t]

   M2  = cv.threshold(M1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   M3  = np.array( 255*np.array( M2 <255, dtype='int'), dtype='uint8')

   return(M3)

############################################################################

def rotateImage(N, angle, center=(0,0)):

   Nflip = flipImage(N)
   M     = Image.fromarray(Nflip).rotate(angle=angle, center=center  )
   N1    = flipImage(np.array(M))

   return(N1)

############################################################################

def WLtoERG(nn, AL, al):   # nn= len(WL)
 
   CL = []
   for ii in range(nn):
      a = [ AL[ii], al[ii], ii]
      CL.append(a)   

   ERG = [CL, 'test']

   return(ERG)

############################################################################

def makeLabel():
 
   global SWO_2D

   ss  = "-SWC"  + "-initEta="+str(SWO_2D.init_eta) + "-J=" + str(SWO_2D.J) +"-nang=" + str(SWO_2D.nang) + "-ll=" + str(SWO_2D.ll) + "-m=" + str(SWO_2D.m) + "-normalization=" + str(SWO_2D.normalization)

   return(ss)

############################################################################

def getRes(OCR, character):
   wn  = np.where( np.array(OCR.l2)==character)
   erg = OCR.rf.predict(OCR.L2)
   L   = []
   for ii in wn[0]:
      L.append([ii, OCR.l2[ii], erg[ii], 1*(OCR.l2[ii] == erg[ii])])     
   for ii in range(len(erg)):
      if erg[ii] == character:
         L.append([ii, OCR.l2[ii], erg[ii], 1*(OCR.l2[ii] == erg[ii])])   
   unique_data = [list(x) for x in set(tuple(x) for x in L)]
   return(unique_data)

############################################################################

def transformMatrix(M):
   imgM   = Image.fromarray(M)
   imgMTF = imgM.filter(ImageFilter.CONTOUR())
   imgMTF = imgMTF.filter(ImageFilter.EMBOSS())
   A      = np.array(imgMTF)

   return(A)

############################################################################

def generateCharacters(generator, character, nn, skewness=True, resize=(80,80)):

   zz = 0
   L = []

   generator.fit = False
   while zz < nn:
      generator.skewing_angle = 0
      if skewness:
         generator.skewing_angle = np.random.randint(10)+ np.random.randint(2)*350
      
      generator.background_type = 1 
      try:
         a       = generator.next();
         zz      = zz+1
         N       = np.array(a[0])[:,:,0]
         A, C    = cut(N)
         L.append( [ C, character] )
      except:
         a = "error"
         print("error")
         zz = zz+1
   return(L)   

############################################################################

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

############################################################################

def getErrorFirstAndSecondKind(OCR, character):

   R        = getRes(OCR, character)
   f1a, f2a = 0, 0
   F1A,F2A  = [], []

   for ii in range(len(R)):
      l = R[ii]
      if l[1] != l[2] and l[1] == character:
         f1a=f1a+1
         F1A.append(l[2])
      if l[1] != l[2] and l[2] == character:
         f2a = f2a+1
         F2A.append(l[1])

   F1A = countEntriesInArrayString(F1A)
   F2A = countEntriesInArrayString(F2A)

   return([R, F1A, F2A])

############################################################################

def countEntriesInArrayString(S):

   ERG = []
   L   = list(set(S))
   for l in L:
       t = list(np.where(np.array(S)==l)[0])
       ERG.append([l, len(t)] )

   return(ERG)

############################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("OCRCalibration.py").read())
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
SWO_2D.init_eta      = 0.4*2*pi
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


from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

#ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö','Ü' ]
##alphabet = list(map( lambda x: str.lower(x) , ALPHABET))
#alphabet = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'q', 'r', 't', 'y', 'ä']
#numbers  = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
#extra    = ['!', '.', ',', '+', '-', ';', ')', '(', '"', '%', 'ß','?','/']


ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'Ä', 'Ö','Ü' ]
alphabet = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'r', 't', 'y', 'ä']

nn             = 500
BIGL           = []

#Lt             = generator.fonts.copy()
FontsOutIndexL = [5, 45,44,32, 73, 86,83,77, 80, 79, 84, 81, 78, 40,36,34,33,30,27,71,70,26,24,66,21,19,65,18,65,64,61,58,16,15,58,14,98,95,55,54,94,10,93,92,89,51,50,9,48,6,47,4,0,1,46]
FontsOutIndexK = [5,52,26,80,77,75,74,73,20,44,41,39,8,32,86,58,59,49,63,83,70,37]
FontsOutIndexJ = [85,83,28,52,25,23,75,49,20,77,44,73,45,97,96,90,86,68,63,59,60,41,37,32,29,8,5,3,62,31,22,56,12,82,42]
FontsOutIndexC = [64, 99, 70, 78, 80, 91, 63, 60,59, 58, 52, 33, 48, 35, 34, 40, 26, 11, 1, 7]

TEST = ['l']

make           = False
if make:
   R     = tqdm( 1*ALPHABET + alphabet) 
   #R     = tqdm(TEST)
   for l in R:
      generator = GeneratorFromStrings(
         [l],
         blur=np.random.randint(4),
         random_blur=True,
         size       = 50,
         text_color =  '#000000'
         ) 
      Lt   = generator.fonts.copy()
      L    = []
      out  = []
      if l == 'k' or l == 'K':
         out = FontsOutIndexK
      if l == 'l' or l == 'L':
         out = FontsOutIndexL 
      if l == 'j' or l == 'J':
         out = FontsOutIndexJ
      if l == 'C' or l == 'c':
         out = FontsOutIndexC

      for jj in range(len(Lt)):
         if jj not in out:
            L.append(Lt[jj])

      generator.fonts = L
      a = generateCharacters(generator, l, nn)
      BIGL.extend(a)


loadModel = True
if loadModel:
   try: 
      a = len(OCR1.L1)
   except:
      OCR1  =  PM.TD("OCR1")   
      #OCR1.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=2-nang=16-ll=3-m=2-normalization=False-25.03.2022-13:58:20')
      #OCR1.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5-J=2-nang=4-ll=3-m=2-normalization=False-26.03.2022-18:05:23')
      OCR1.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5132741228718345-J=2-nang=16-ll=3-m=2-normalization=False-28.03.2022-00:19:38')
      OCR1.evaluate(n=2000, write=False, crossVal=False, dict=False, pri=False)
   
   try:
      a = len(OCR2.L1)
   except:
      OCR2  =  PM.TD("OCR2")  
      OCR2.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG-SWC-initEta=2.5132741228718345-J=2-nang=4-ll=3-m=2-normalization=False-27.03.2022-22:35:21')
      OCR2.evaluate(n=2000, write=False, crossVal=False, dict=False, pri=False)

makeCalcs    = False
makeOCRModel = False
if makeCalcs:

   CL   = list(map(lambda x: enlargeGreyMatrix( x[0], 100,100), BIGL ))
   al   = list(map(lambda x: x[1], BIGL))
   WL1  = MISC.makeIt(CL, SWO_2D, "rf") 
   ERG1 = WLtoERG(len(al), WL1, al)
   
   #SWO_2D.J    = 2 
   #SWO_2D.nang = 8
   #SWO_2D.ll   = 2
   #SWO_2D.jmax = SWO_2D.J
   #WL2  = MISC.makeIt(CL, SWO_2D, "rf")  

   if makeOCRModel:
      ss    = makeLabel()
      dst   = MISC.saveIt(ERG1, "data/garbage/testERG"+ ss)
      OCR  =  PM.TD("bB")   
      OCR.loadData('/home/markus/anaconda3/python/data/garbage/', 'testERG' + ss + '-' + dst)
      OCR.evaluate(n=2000, write=False, crossVal=False, dict=False, pri=False)



#print(OCR1.rf.predict(WL1))
#print(OCR2.rf.predict(WL2))

#R, F1A, F2A = getErrorFirstAndSecondKind(OCR1, 'l')


makeTest = False
if makeTest:
   character      = 'l I'
   generator = GeneratorFromStrings(
       [character],
       blur=np.random.randint(4),
       random_blur=True,
       size       = 50,
       skewing_angle = np.random.randint(20)+ np.random.randint(2)*340,
       text_color =  '#000000',
   ) 

   Lt                 = generator.fonts.copy()
   FontsOutIndexTotal = 0*FontsOutIndexL + 0*FontsOutIndexK + 1*FontsOutIndexJ

   E  = []

   for jj in range(0, len(Lt)):
      if jj not in FontsOutIndexTotal: 
         font = Lt[jj]
         generator.fonts = [ font]
         a    = generateCharacters(generator, character, 1, False)
         img  = Image.fromarray(a[0][0])
         draw = ImageDraw.Draw(img)
         draw.text( (2,2), str(jj), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=12)) 
         E.append(img)

   showAllImages(E)



IndexIJStrange = [70,64,61,78,14,47,40,39]


