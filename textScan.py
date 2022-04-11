import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont, ImageFilter
import sys, subprocess
sys.path.append('/home/markus/anaconda3/python/development/modules')
from tqdm import tqdm
from sklearn import metrics
from datetime import datetime 
import cv2 as cv
import pytesseract
from pytesseract import Output
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
import pandas as pd

from craft_text_detector import Craft

import pymysql as mysql    # pip install mysql-connector-python
import mysql.connector
from sqlalchemy import create_engine
import pymysql 
pymysql.install_as_MySQLdb()


import misc_v9 as MISC
import dataOrganisationModule_v3 as dOM
import morletModule_v2 as MM  
import DFTForSCN_v7 as DFT
import scatteringTransformationModule_2D_v9 as ST
import predictionModule_v2 as PM
 

###################################################################################################################

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###################################################################################################################

class point:
   def __init__(self, x,y,name='point' ):   
      self.x = x
      self.y = y

class kernel:
   def __init__(self, name='kernel' ):
      self.name    = name   
      self.boxBlur = (1 / 9.0) * np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
      self.sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
 
class JPG:
   def __init__(self, path, filename, name='JPG' ):
      self.path     = path 
      self.filename = filename   
      self.name     = name   

   def open(self):
      img = Image.open(self.path + self.filename).convert('L')
      return(img)

class radius:
   def __init__(self, name='radius' ):
      self.name = name   
      self.blur = 1
      self.edge = 1

class sigma:
   def __init__(self, name='sigma' ):
      self.name = name   
      self.blur = 0.75
  

class imageFilter:
   def __init__(self, name='imageFilter' ):
      self.name   = name   
      self.radius = radius()
      self.sigma  = sigma()

class boxFilter:
   def __init__(self, name='boxFilter' ):
      self.name = name  
      self.bx   = [0, 50]
      self.by   = [0, 50] 

class matrixFilter:
   def __init__(self, name='matrixFitler' ):
      self.name = name   
      self.lb   = 0.48 #0.57
      self.ub   = 0.53 #0.65      

class resize:
   def __init__(self, name='resize' ):
      self.name = name   
      self.fxG  = 3000
      self.fyG  = 3000

class preprocessing:
   def __init__(self, name='preprocessing' ):
      self.name               = name   
      self.frontierBlackWhite = 130
      self.imageFilter        = imageFilter()

class annotation:
   def __init__(self, name='annotation' ):
      self.name = name   
      self.a    = []
      self.e    = []
      self.i    = []
      self.t    = []
      self.s    = []

class LABEL:
   def __init__(self, path, filename,  name='LABEL' ):
      self.name          = name
      self.path          = path
      self.filename      = filename
      self.JPG           = JPG(path, filename)
      self.boxFilter     = boxFilter()
      self.matrixFilter  = matrixFilter()
      self.kernel        = kernel()
      self.resize        = resize()
      self.preprocessing = preprocessing()
      self.annotation    = annotation()
      self.imageFilter   = imageFilter()

   ###################################################################################################################

   def getPercentageValue(self, ht, lb, ub):
      lbW          = 0
      found       = False
      while lbW <= 257 and not(found):
         lbW = lbW +1
         erg = sum(ht[lbW: ])
         if lb <= erg <= ub:
            found=True

      return(lbW)

   ###################################################################################################################

   def makeHistoPerc(self, M):
      h,b   = np.histogram(M, bins=list(range(0,257)))
      ht    = np.round(h/np.prod(M.shape),3)
   
      return(ht)

   ###################################################################################################################

   def applyFilterBoxMatrix(self, boxL, kernel, display=False, overRidelBW = False):
      NL, ML, IMGL = [],[], []
      
      for ii in range(len(boxL)):
         box         = boxL[ii]
         x1,y1,x2,y2 = box
         M           = self.JPG.M2[y1:y2, x1:x2]
         ht          = self.makeHistoPerc(M[:,:,0])
         if overRidelBW:   
            self.preprocessing.frontierBlackWhite = self.getPercentageValue(ht, self.matrixFilter.lb, self.matrixFilter.ub)   
          
         N_dn        = cv.filter2D(src= M, ddepth= -1, kernel= kernel)
         N2          = np.array( 255*np.array( N_dn > self.preprocessing.frontierBlackWhite, dtype= 'int'), dtype= 'uint8');
 
         NL.append(N2)
         ML.append(M)

         if display:
            img  = Image.fromarray(N2)
            draw = ImageDraw.Draw(img)
            draw.text( (3,3), str(ii) ,fill=(128,0,120), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=9))
            #img.show()   
            IMGL.append(img)

      return([NL, ML, IMGL])

   ###################################################################################################################

   def preProcessingImage(self, Ct, fx, fy):
      M2     = cv.resize(Ct, None, fx = fx, fy = fy, interpolation = cv.INTER_CUBIC)
      dst    = cv.fastNlMeansDenoisingColored(src = M2, dst = None, h = 5, hColor = 10, templateWindowSize = 7, searchWindowSize = 15)      
      dst_dn = cv.filter2D(src = dst, ddepth = -1, kernel = self.kernel.sharpen)
      C      = np.array( 255*np.array(dst_dn > self.preprocessing.frontierBlackWhite, dtype='int'), dtype='uint8')
      self.C = C
      self.dst = dst
      self.dst_dn = dst_dn

      ny     = Wimage.from_array(C)
      ny.edge(self.imageFilter.radius.edge)
      ny.blur(radius = self.imageFilter.radius.blur, sigma=self.imageFilter.sigma.blur)
      A1     = np.array(ny)[:,:,0]
      B1     = np.array( 255*np.array(A1 < 1, dtype='int'), dtype='uint8')   
      self.A1 = A1
      self.B1 = B1

      return([B1, M2])

   ###################################################################################################################

   def calcAll(self):
      imgOrgt            = self.JPG.open()
      A                  = np.array(imgOrgt)
      #A                  = cv.threshold(A, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
      s                  = A.shape
      if len(s)==3:
         imgOrg, Bt         = matrixToRGBImage(A[:,:,0])
      else:
         imgOrg, Bt         = matrixToRGBImage(A)

      
      height, width      = imgOrgt.height, imgOrgt.width
      B1, self.JPG.M2    = self.preProcessingImage(Bt, fx = self.resize.fxG/width, fy = self.resize.fyG/height)

      self.B1            = B1
      self.boxL          = getBoxFromContours(B1, progressbar=False, hr=0)
      self.boxL1         = boxFilterByInterval(self.boxL, self.boxFilter.bx, self.boxFilter.by, progressbar=False)
      self.boxL2         = removeBoxContainedInOtherBox(self.boxL1, progressbar=False)
      self.title         = "bx:" + str(self.boxFilter.bx) + " by:" + str(self.boxFilter.by) + " fxG:" + str(self.resize.fxG) + " fyG:" + str(self.resize.fyG) + " lbW:" + str(self.preprocessing.frontierBlackWhite) 
      self.title         = self.title  + " re:" + str(self.imageFilter.radius.edge) + " rb:" + str(self.imageFilter.radius.blur) + " si:" + str(self.imageFilter.sigma.blur)
  
   ###################################################################################################################

   def makeM2(self):
      imgOrgt            = self.JPG.open()
      A                  = np.array(imgOrgt)
      s                  = A.shape
      if len(s)==3:
         imgOrg, Bt         = matrixToRGBImage(A[:,:,0])
      else:
         imgOrg, Bt         = matrixToRGBImage(A)

      height, width      = imgOrgt.height, imgOrgt.width
      self.JPG.M2        = cv.resize(Bt, None, fx = self.resize.fxG/width, fy = self.resize.fyG/height, interpolation = cv.INTER_CUBIC)


###################################################################################################################

def drawBoxes(boxL, sizeMat, img='', font='Roboto-Bold.ttf', sizeFont=12, makeDenotations=True, width=1, title='', drawTitle=True, progressbar=False):


   boxL.sort(key=lambda x: x[0])

   if img=='':
      img    = Image.new(mode="RGB", size=sizeMat, color=(255,255,255))
   draw   = ImageDraw.Draw(img)
   R = range(len(boxL))
   if progressbar: 
      R      = tqdm(range(len(boxL)))
      R.set_description('drawBoxes')

   for ii in R:
      box            = boxL[ii]
      x1,y1,x2,y2    = box
      #mx             = int( (x1+x2)/2)
      #my             = int( (y1+y2)/2)
   
      draw.rectangle(box, width=width, outline="red") 
      if makeDenotations:
         draw.text( (x1,y1), str(ii) ,fill=(0,0,0), font=ImageFont.truetype(font=font, size=sizeFont))
         
   
   if drawTitle:
      draw.text( (20,20), title ,fill=(0,0,0), font=ImageFont.truetype(font=font, size=30))

   return(img )

###################################################################################################################

def matrixToRGBImage(M):

   size = ( M.shape[1], M.shape[0])
   img  = Image.new(mode="RGB",size=size, color=(255,255,255))
   A    = np.array(img)
   A[:,:,0] = M
   A[:,:,1] = M
   A[:,:,2] = M

   img = Image.fromarray(A)

   return([img, A]) 

###################################################################################################################

def boxFilterByInterval(boxL, bx, by, progressbar=True):

   boxL_fil = []
   R        = tqdm(boxL)
   R.set_description('boxFilterByInterval')
   boxL_fil   = list(filter(lambda x: bx[0]  <= x[2]-x[0] <= bx[1] and by[0] <= x[3] - x[1] <= by[1], R))
   boxL0      = unique(boxL_fil)
   return( boxL0)

###################################################################################################################

def removeBoxContainedInOtherBox(boxL, progessbar=False):

   boxL     = unique(boxL)
   R        = list(range(len(boxL)))
   if progessbar:
      R        = tqdm(R)
      R.set_description('removeBoxContainedInOtherBox - 1')
   L        = []
  
   for ii in R:
      x1,y1,x2,y2 = box1 = boxL[ii]
      l           = list(filter(lambda x: (x1 <= x[0] <= x[2] <= x2) and (y1 <= x[1] <= x[3] <= y2), boxL))
      l.remove(box1)
      L.extend(l) 

   L       = unique(L)
   R       = boxL
   if progessbar:
      R       = tqdm(R)
      R.set_description('removeBoxContainedInOtherBox - 2')

   ERG   = list(filter(lambda x: x not in L, R))

   return( ERG)

###################################################################################################################

def getBoxFromContours(B1, progressbar=True, hr=0):

   contours, hierachy = cv.findContours(B1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   R       = range(len(contours))

   if progressbar:
      R       = tqdm(R)
      R.set_description('getBoxFromContours')
   boxL    = []

   for ii in R:
      c                                   = contours[ii]
      Next, Previous, First_Child, Parent = hierachy[hr,ii,:]
      x, y, w, h                          = cv.boundingRect(c)
      box                                 = [x, y, x+w, y+h]
      boxL.append(box)
         
   return(boxL)

###################################################################################################################

def rotateImage(N, angle, center=(0,0)):

   Nflip = flipImage(N)
   M     = Image.fromarray(Nflip).rotate(angle=angle, center=center  )
   N1    = flipImage(np.array(M))

   return(N1)

###################################################################################################################

def flipImage(img, t=0):

   M1  = np.array(img)
   if len(M1.shape)==3:
      M1 = M1[:,:,t]

   M2  = cv.threshold(M1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   M3  = np.array( 255*np.array( M2 <255, dtype='int'), dtype='uint8')

   return(M3)

###################################################################################################################

def makeWhiteUint8( n,m ):
   R = np.array( 255*np.ones( (n,m)), dtype='uint8')
   return(R)

###################################################################################################################

def enlargeGreyMatrix( M, n, m):
   C      = makeWhiteUint8( n,m )
   nt, mt = M.shape
   if nt <= n and mt <= m:
      dn, dm = int( 0.5*(n-nt)), int(0.5*(m-mt))
      C[dn:dn+nt, dm:dm+mt] = M

   return(C)

###################################################################################################################

class ALLABS:
   def __init__(self, name='all labels' ):
      self.name = name   
   
   ###################################################################################################################

   def makeMatricesFromAnnotations(self, letter, NL):
      MATL_N = []
      MATL_M = []
      IL     = []
      for name in NL:
         O          = getattr(self, name.split('.')[0])
         P          = O.annotation
         Q          = getattr(P, letter)
         boxL       = list(map( lambda x: O.boxL2[x], Q))
         N, M, IMGL = O.applyFilterBoxMatrix(boxL, O.kernel.boxBlur, True, True)
         ERG_N      = list(map(lambda x: [x, letter], N))
         ERG_M      = list(map(lambda x: [x, letter], M))

         IL.extend(IMGL) 
         MATL_N.extend(ERG_N)
         MATL_M.extend(ERG_M)

      return([MATL_N,MATL_M, IL])

###################################################################################################################

def unique(L):
   return(  [list(x) for x in set(tuple(x) for x in L)])

###################################################################################################################

def cart2pol(P):  
   x, y = P
   rho  = np.sqrt(x**2 + y**2)
   phi  = np.arctan2(y, x)
   return(rho, phi)

################################################################################################

def pol2cart(C):
   rho, phi = C
   x = round( rho * np.cos(phi), 0)
   y = round( rho * np.sin(phi), 0)
   return( [x, y])

################################################################################################

def pBL(M2, boxL):
   imgBIGL, _  = matrixToRGBImage(M2)
   imgBIGL     = drawBoxes(boxL = boxL, sizeMat = (3000,3000), img=imgBIGL, width=1, progressbar = True, makeDenotations=False)
   imgBIGL.show()

   return(imgBIGL)

################################################################################################

def makeWhiteUint8( n,m ):
   R = np.array( 255*np.ones( (n,m)), dtype='uint8')
   return(R)

###################################################################################################################

def transformBoxes2(boxL1, r, origin=(0,0), transformToPolygon=True):

   a, b              = origin[0], origin[1]
   if transformToPolygon==False:
      L                 = list(map(lambda x: [(x[0][0]-a, x[0][1]-b), (x[1][0]-a, x[1][1]-b), (x[2][0]-a, x[2][1]-b), (x[3][0]-a, x[3][1]-b)] , boxL1))
   else:
      L                 = list(map(lambda x: [(x[0]-a, x[1]-b), (x[2]-a, x[1]-b), (x[2]-a, x[3]-b), (x[0]-a, x[3]-b)] , boxL1))

   boxL1PC           = list(map(lambda x: [ cart2pol(x[0]), cart2pol(x[1]), cart2pol(x[2]), cart2pol(x[3]) ], L))
   boxL1PCt          = list(map(lambda x: [ (x[0][0], x[0][1]+r), (x[1][0], x[1][1]+r), (x[2][0], x[2][1]+r), (x[3][0], x[3][1]+r) ], boxL1PC))
   boxL1t            = list(map(lambda x: [ pol2cart(x[0]), pol2cart(x[1]), pol2cart(x[2]), pol2cart(x[3])], boxL1PCt))
   boxL2t            = list(map(lambda x: [ ( x[0][0]+a, x[0][1]+b), (x[1][0]+a, x[1][1]+b), (x[2][0]+a, x[2][1]+b), (x[3][0]+a, x[3][1]+b)], boxL1t))

   return(boxL2t)

###################################################################################################################

def shiftPoint(P, origin):
   a, b = origin[0], origin[1]
   Q    = (P[0]-a, P[1]-b)
   return(Q)

###################################################################################################################

def rotatePoint(P, angle, origin):
   a, b = origin[0], origin[1]
   r    = angle*(pi/180)
   Q    = shiftPoint(P, (a,b))
   C    = cart2pol(Q)
   Ct   = (C[0], C[1] + r)
   Qt   = pol2cart(Ct)
   Qtt  = shiftPoint(Qt, (-a, -b))
 
   return(Qtt)

###################################################################################################################
# poly = (x1,y1), (x2,y1), (x2,y2), (x1,y2)
def polygonToBox(poly):
   P1,P2,P3,P4 = poly
   box         = [ P1[0], P1[1], P3[0], P3[1]]
   return(box)

###################################################################################################################

def boxToPolygon(box):
   x1,y2,x2,y2 = box
   poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
   return(poly)

###################################################################################################################

def makeTestMatrix(M2,angle, xb=[0,3000], yb=[0,3000], a=0, b=3000):

   M2rot          = rotateImage(M2.copy(), angle)
   M2T            = cv.threshold(M2rot, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   imgM2T         = Image.fromarray(M2T)
   imgM2TF        = imgM2T.filter(ImageFilter.CONTOUR())
   imgM2TF        = imgM2TF.filter(ImageFilter.EMBOSS())
   At             = np.array(imgM2TF)

   Bt             = At[a:b, :]
   M2t            = M2rot[a:b, :]
   boxL0          = getBoxFromContours(Bt, progressbar=False, hr=0)
   boxL1          = boxFilterByInterval(boxL0, xb, yb, progressbar=True)

   return([Bt, M2t, boxL1])

###################################################################################################################

def checkIntersectionOfLines(boxL, a=0, b=3000, stepSize=5, progressbar=True):
   anzAlt = 0
   LL     = []
   R = range(a,b,stepSize)
   if progressbar:
      R = tqdm(R)

   for zz in R:
      WW = list(filter( lambda x: max(list(map(lambda y: y[1], x))) <= zz, boxL))
      V = list(filter( lambda x: min(list(map(lambda y: y[1], x))) <= zz <= max(list(map(lambda y: y[1], x))), boxL))
      if len(WW)> anzAlt and len(V)==0:
         anzAlt = len(WW)
         LL.append(zz)

   return(LL)

###################################################################################################################

def plotLineSeeking(boxL1, angle, mdict, orgin= (1500, 1500), width=3000, height=3000):

   W                     = makeWhiteUint8( width, height )
   r                     = angle*(pi/180)
   imgW, _               = matrixToRGBImage(W)
   D                     = ImageDraw.Draw(imgW)
   #m1                   = Dict[str(angle), str(startList[jj2]), str(wsL[ii2])]
   for ii in range(len(mdict)):
      a,b, nn, ll = mdict[ii]
      D.line( (0, a, width,a), width=3, fill=128)
      D.text( (0, a), str(nn) ,fill=(0,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))
      D.text( (width/2, a), str(nn) ,fill=(0,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))
      D.text( (width-100, a), str(nn) ,fill=(0,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))

   boxL1t  = transformBoxes2(boxL1, r, origin )
   D.text( (30,10), str(angle),  fill=(0,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))
   for ii in range(len(boxL1)):
      D.polygon(boxL1[ii], outline='blue',fill ="#ffce30" ) 

   return(imgW)

###################################################################################################################

def removeSkewness(aL, boxL1, stepSize=20, origin=(1500,1500), a=0, b=3000):

   erg = []
   R   = tqdm(aL)
   R.set_description('removeSkewness')

   for angle in R:
      r        = angle*(pi/180)
      boxL1t   = transformBoxes2(boxL1, r, origin )
      LL       = checkIntersectionOfLines(boxL1t, a=a, b=b, stepSize=stepSize, progressbar=False)
      erg.append([angle, r, len(LL)])

   m = max(list(map(lambda x:x[2] , erg)))
   l = list(filter(lambda x: x[2]==m, erg))
   l.sort(key=lambda x: abs(x[0]))

   angle = l[0][0]

   return(angle)

###################################################################################################################

def makeBoxList(BOXLt):
   BOXL = []
   for ii in range(len(BOXLt)):
      boxt      = BOXLt[ii]
      xl, yl    = list(boxt[:,0]), list(boxt[:,1])
      box       = [ min(xl), min(yl), max(xl), max(yl)]
      BOXL.append(box)

   return(BOXL)

###################################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("textScan.py").read())
#
##   

######################################### initial part ############################################################
   
path      = '/home/markus/Documents/grundbuchauszuegeAnnotation/'
pathData  = '/home/markus/anaconda3/python/data/'
makeCalcs = False

NL        = ['muenchen-trudering-bestandsverzichnis.jpg',
             'bonn-beuel-abteilung-1.jpg'               , 
             'bonn-beuel-abteilung-2.jpg'               , 
             'bonn-beuel-abteilung-3.jpg'               ,
             'abteilung-1-schreibmaschineSchiefGescanntAberGutLesbar.jpg',
             'leipzig-leipzig-abteilung-1.jpg'          ,
             'siegburg-obermenden-abteilung-1.jpg'      ,
             'siegburg-obermenden-abteilung-3.jpg'      ,
             'muenchen-muenchen-abteilung-3.jpg'        ,
             'abteilung-1-schreibmaschineGutLesbar-1.jpg',
             'goslar-bündheim-bestandsverzeichnis.png',
             'goslar-bündheim-abteilung-2.png' ]


try:
   a= BOXL.name
except:
   print("except")
   BOXL   = ALLABS()
   for name in tqdm(NL): 
      O      = LABEL(path, name)
      O.makeM2()
      #O.calcAll()
      O.size = (O.JPG.M2.shape[1], O.JPG.M2.shape[0])  
      setattr(BOXL, name.split('.')[0], O)




MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = O.JPG.M2[:,:,0]
adaptMatrixCoef      = tuple([100, 100])
Ct                   = MAT.downSampling(C1, 4)                
dx,dy                = 0.15, 0.15
SWO_2D               = MM.SWO_2D(Ct, round(Ct.shape[1]*0.5*dx,3), round(Ct.shape[0]*0.5*dy,3))
SWO_2D.init_eta      = 0.4*2*pi
SWO_2D.kk            = 1
SWO_2D.sigma         = mat([[SWO_2D.kk,0],[0,SWO_2D.kk]])
SWO_2D.J             = 2
SWO_2D.nang          = 8
SWO_2D.ll            = 3
SWO_2D.jmax          = SWO_2D.J
SWO_2D.m             = 2
SWO_2D.normalization = False   # (m=2 wird mit m=1-Wert normalisiert)
SWO_2D.onlyCoef      = True    # für debugging Zwecke auf False setzen 
SWO_2D.allLevels     = False   # wenn True dann werden nur Werte für m=2, ansonsten auf m=1 geliefert (DS enthält trotzdem alles für debugging Zwecke)



############################################  main  ###############################################################


nn         = 2
O          = getattr(BOXL, NL[nn].split('.')[0]) 
O.preprocessing.frontierBlackWhite = 200
try:
   a = getattr(O, 'calcAll')
except:
   O.calcAll()

O.makeM2()

M2                = O.JPG.M2[:,:,0]
aL                = np.arange(-5, 5.5, 0.5)
origin            = (1500,1500)
width, height     = 3000,3000

Bt, M2t, boxL1    = makeTestMatrix(M2, 0, xb=[10,50], yb=[10, 50])
angle             = removeSkewness(aL, boxL1)

N                 = rotateImage(M2t, angle=-angle, center=origin)
N[999:1001, :]    = 0
N[1999:2001, :]   = 0
M2[999:1001, :]   = 0
M2[1999:2001, :]  = 0

Image.fromarray(N).show()
Image.fromarray(M2).show()


makeCraft         = True
output_dir        = 'outputs/'
craft             = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

if makeCraft:
   try: 
      d = len(prediction_result.keys())
   except:
      prediction_result = craft.detect_text(N)
      craft.unload_craftnet_model()
      craft.unload_refinenet_model()

      img, A  = matrixToRGBImage(N)
      draw    = ImageDraw.Draw(img)

      BOXLt   = prediction_result['boxes']
      CBOX    = makeBoxList(BOXLt)

      imgMB   = drawBoxes(boxL = CBOX, sizeMat = (3000,3000), sizeFont=25,img=img, width=3, progressbar = True, makeDenotations=True)
      imgMB.show()

      #BIGL0 = getComplement(boxL1, CBOX)












