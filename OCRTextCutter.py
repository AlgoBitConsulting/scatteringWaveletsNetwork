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
#import pandas as pd

from craft_text_detector import Craft


###################################################################################################################

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###################################################################################################################

#name, bx, by, fxG=3000, fyG=3000, lbW=130, radius_edge=1, radius_blur=1, sigma =  0.75):


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
   #R = list(range(len(boxL)))
   #if progressbar:
   #   R = tqdm(R)
   #   R.set_description('boxFilterByInterval')
   #for ii in R:
   #   x1,y1,x2,y2 = box = boxL[ii]
   #   
   #   if bx[0]  <= x2-x1 <= bx[1] and by[0] <= y2 - y1 <= by[1]: 
   #      boxL_fil.append(box)

   R       = tqdm(boxL)
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

def showAllImages(IL):
   for img in IL:
      img.show()

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


def flipImage(img, t=0):

   M1  = np.array(img)
   if len(M1.shape)==3:
      M1 = M1[:,:,t]

   M2  = cv.threshold(M1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   M3  = np.array( 255*np.array( M2 <255, dtype='int'), dtype='uint8')

   return(M3)

###################################################################################################################

def makeBoxesFromCut( N, l, col=125):

   erg   = []
   a     = [0]
   a.extend(l)
   a.append(N.shape[1])
   for ii in range(len(a)-1):
      T   = N[:, a[ii]:a[ii+1]]   
      B   = cutOutEmptyPartsOfMatrix(T)  
      erg.append(B)
   
   for ii in range(len(l)):
      N[:, l] = col

   return([N, erg])


###################################################################################################################

def getBoxes(N):

   boxL2   = getBoxFromContours(N)
   xmed    = np.median(list(map(lambda x: x[2]-x[0], boxL2)))
   ymed    = np.median(list(map(lambda x: x[3]-x[1], boxL2)))

   return([boxL2, xmed, ymed])

###################################################################################################################

def filterBoxes(boxL2, xb, yb):

   boxL2t  = list(filter( lambda x: xb[0] <= x[2]-x[0] <= xb[1] and yb[0] <= x[3]-x[1] <= yb[1], boxL2))

   return(boxL2t)

###################################################################################################################   
 
def getAlpha(boxL2):
   
   boxL2t     = boxL2.copy()
   boxL2t.sort(key=lambda x: x[2])
   lbox, rbox = boxL2t[0], boxL2t[len(boxL2t)-1]
   A = point(lbox[2], lbox[3])
   B = point(rbox[2], rbox[3]) 
   C = point(rbox[2], lbox[3])
   a = C.x - A.x
   b = C.y - B.y
   c = sqrt( a**2 + b**2)
   alpha = np.arcsin( b/c)
   angle = alpha/(pi/180)

   return([alpha, angle, A, B, C])

###################################################################################################################

def drawTriangle(N, A,B,C):

   img = Image.fromarray(N)
   draw = ImageDraw.Draw(img)
   draw.line(( A.x, A.y, B.x, B.y), fill=128)
   draw.line(( C.x, C.y, B.x, B.y), fill=128)
   draw.line(( A.x, A.y, C.x, C.y), fill=128)
   img.show()

   return(img)
   
###################################################################################################################

def removeNoise(N, boxL2t):

   Nt = N.copy()

   if len(boxL2t) >0:
      ymax = max(list(map( lambda x: x[3]-x[1], boxL2t)))+8
      S = N.sum(axis=1)/(N.shape[1]*255)
      m, erg = 1, []
      for ii in range(len(S)-ymax):
         r = sum(S[ii: ii+ymax])/ymax
         if r < m:
            erg.append([r, ii])

      if len(erg)>0:
         d = min( list(map(lambda x: x[0] ,erg)))
         r = list(filter( lambda x: x[0]==d, erg))
         Nt = N[r[0][1]: r[0][1]+ymax, :]

   return(Nt)

###################################################################################################################

def graphInImage(N):

   n      = N.sum(axis=0)/(N.shape[0]*255)
   img, A = matrixToRGBImage(N)
   draw   = ImageDraw.Draw(img)

   xy   = []
   for ii in range(len(n)):
      xy.extend([ii, (1-n[ii])*N.shape[0]])
  
   draw.polygon(xy, outline='red')

   return(img)

###################################################################################################################

def removeSkweness(N, xb=[15,70], yb=[30,70], drawTri=False):

   boxL2                 = []
   alpha, angle          = 0,0
   N1,N2                 = N,N

   boxL2t, xmed, ymed    = getBoxes(N) 
   boxL2                 = filterBoxes(boxL2t, xb, yb)
   if len(boxL2) >0:
      alpha, angle, A, B, C = getAlpha(boxL2)
      if drawTri:
         triImg                = drawTriangle(N, A,B,C)
      N1                    = rotateImage(N, -angle)
      
   return([N1, angle,  boxL2t, boxL2])

###################################################################################################################

def intersectIntervals(x1,x2, s1,s2):

   if x2 <= s1 or s2 <= x1:
      return([-1,-1])
   if x1 <= s1 <= s2 <= x2:
      return( [s1,s2])
   if s1 <= x1 <= x2 <= s2:
      return([x1,x2])
   if s1 <= x1 <= s2 and x1<= s2 <= x2:
      return([s2,x1])
   if x1 <= s1 <= x2 and s1 <= x2 <= s2:
      return([x2,s1])

################################################################################################

def intersectionOfBoxes(box1, box2):

   x1,y1,x2,y2 = box1
   s1,t1,s2,t2 = box2

   u1,u2       = intersectIntervals(x1,x2,s1,s2)
   v1,v2       = intersectIntervals(y1,y2,t1,t2)   

   if u1==u2==-1 or v1==v2==-1:
      return((0,0,0,0))
   else:
      return( (u1,v1,u2,v2))

###################################################################################################################

def boxArea(box):
   x1,y1,x2,y2 = box
   erg = abs( (x1-x2)*(y1-y2))
   return(erg)

###################################################################################################################

def unionOfBoxes(box1, box2):

   x1,y1,x2,y2 = box1
   s1,t1,s2,t2 = box2

   nb = ( min(x1,x2,s1,s2), min(y1,y2,t1,t2), max(x1,x2,s1,s2), max(y1,y2,t1,t2))

   return(nb)

################################################################################################

def isContainedIn(box1, box2):

   x1,y1,x2,y2 = box2
   v1,w1,v2,w2 = box1
   contained   = False 
   if x1 <= v1 <=  v2 <= x2 and y1 <= w1 <= w2 <= y2:
      contained=True

   return(contained)

################################################################################################

def getAllBoxesContainedInGivenBox(box1, boxL, overlapping=False, overlappingFactor = 0.7):

   contained   = []
   overlapping = []
   x1,y1,x2,y2 = box1
   
   for ii in range(len(boxL)):
      v1,w1,v2,w2 = box2 = boxL[ii]  
      if isContainedIn(box2, box1):
         #if x1 < v1 <  v2 < x2 and y1 < w1 < w2 < y2:
         contained.append(box2)
      if overlapping:
         box3 = intersectionOfBoxes(box2, box1)
         if boxArea(box3)/boxArea(box2) >= overlappingFactor:
            overlapping.append(box2)
   
   return([contained, overlapping])

################################################################################################

def makeVerticalLinesWithWindow(N, ws,fak=0.9, display=False ):

   zz = 0
   erg = []
   while zz < N.shape[1]-ws:
      b = N[:, zz:zz+ws]
      r = sum(b.sum(axis=0)/(255*N.shape[0]))/N.shape[1] 
      erg.append(round(r,4))
      zz = zz+1

   a = erg > max(erg)*fak
   zz,l = findVerticalLines(N, a, st=0, col=255)

   try:
      l.remove(0)
   except:
      ff = 3

   Gt, ergt    = makeBoxesFromCut( N, l) 

   if display:
      Image.fromarray(Gt).show()

   return([Gt, ergt, zz, l])

################################################################################################

def predOver(MATL, orginal):

   global rf_OCR
   global SWO_2D

   ERG        = []

   a          = list(map( lambda x: enlargeGreyMatrix(x[0], 100,100), MATL))
   WLt        = MISC.makeIt(a, SWO_2D, "rf") 
   predicted  = addStr(rf_OCR.predict(WLt))
   P          = rf_OCR.predict_proba(WLt)

   for nn in range(len(orginal)):
      char_pred  = predicted[nn]
      char_org   = orginal[nn]
      #nd = list(rf.classes_).index('d')

      K = [ list(rf_OCR.classes_), list(P[nn, :])]
      A = pd.DataFrame( tp(K), columns=['label', 'probability'])   
      A = A.sort_values(by='probability', ascending=False)
      A = A.reset_index()
      A.index +=1

      
      try:
         rang  = A.label.tolist().index(char_org)+1
         prob  = A.probability[rang]
         prob1 = A.probability[1]
      except:
         rang= prob=prob1 = -1   

      ERG.append([char_pred, rang, prob, prob1])

   E = pd.DataFrame( tp(ERG), columns=list(orginal))

   return(E)

################################################################################################

def getIntersection(box1, boxL, glx, gly):

   x1,y1,x2,y2 = box1   
   xl,xo       = x1-glx, x2+glx
   yl,yo       = y1-gly, y2+gly
   boxLF       = list(filter( lambda x: (xl <= x[0] <= xo or xl <= x[2] <= xo) and ( yl <= x[1] <= yo or yl <= x[3] <= yo), boxL)) 

   return(boxLF)

################################################################################################

def unionOfAllBoxes(boxL, progressbar, glx=0, gly=0):

   boxLC    = [] 
   R        = list(range(len(boxL)))
   L        = R.copy()
   if progressbar:
      R        = tqdm(R)
      R.set_description('unionOfAllBoxes')

   for ii in R:
      box1         = boxL[ii]
      ints         = getIntersection(box1, boxL, glx, gly)
      ubox         = x1,y1,x2,y2 = min(list(map(lambda x: x[0], ints))), min(list(map(lambda x: x[1], ints))), max(list(map(lambda x: x[2], ints))), max(list(map(lambda x: x[3], ints)))

      boxLC.append(list(ubox))
 
   boxL0      = unique(boxLC)

   return( boxL0)


################################################################################################

def unique(L):
   return(  [list(x) for x in set(tuple(x) for x in L)])

################################################################################################

def stepThroughImage(Mt, boxL, windowSize, stepSize, start=0, lb=20):

   erg  = []
   M    = Mt.copy()
   BW   = []
   Wcut = []
   for ii in range(start, M.shape[0]-windowSize, stepSize):
      W = list(filter( lambda x: ii <= x[1] <= x[3] <= ii+windowSize, boxL))
      erg.append([ii, ii+ windowSize, len(W)])
      BW.append(W)
 
   m   = list(map( lambda x: x[2], erg))
   cut = []
   for ii in range(1,len(m)-1):
      if m[ii-1] < m[ii] and m[ii+1] <= m[ii] and m[ii]>lb:
         cut.append(erg[ii])
         Wcut.append(BW[ii])  
         a,b,c = erg[ii]
         M[a-5:a+5, :] = 0 
         M[b-1:b+1, :] = 0 

   return([M, erg, m, cut, Wcut])

################################################################################################

def getAverageHeightOfTextLine(wsL, M2, boxLAt, startList):

   ERG = []
   A   = []
   for start in startList:
      tt = []
      for ws in wsL:
         M, erg, m, cut, wcut = stepThroughImage(M2.copy(), boxLAt, windowSize=ws, stepSize=int(0.5*ws), start=start)
         tt.append([ws, len(cut)])
  
      m = max(list(map(lambda x: x[1], tt)))
      l = list(filter(lambda x: x[1] == m, tt))
      r = np.mean(list(map(lambda x: x[0], l)))
      A.append([r, start])
      ERG.append(r)

   return([np.mean(ERG), A])

################################################################################################

def myShow(nn, L, A, progressbar=False, makeDenotations=False):

   imgA, A    = matrixToRGBImage(A)
   imgBoxesL  = drawBoxes(L[nn], imgA.size, imgA ,font='Roboto-Bold.ttf', sizeFont=10, makeDenotations=makeDenotations, width=1, title='', progressbar=progressbar)
   #imgBoxesL.show()

   return(imgBoxesL)
   
################################################################################################

def checkPage(A, bx=[10,50], by=[10,50], r1=2, r2=2, progressBar=False):         #dünnt Boxen aus

   boxL0      = getBoxFromContours(A, progressbar=False, hr=0)
   boxL       = boxFilterByInterval(boxL0, bx, by, progressbar=progressBar)

   lalt       = len(boxL)
   ii         = 0

   while ii <= r1:
      boxL = unionOfAllBoxes(boxL, progressbar=progressBar)
      ii = ii+1
      if len(boxL) ==lalt:
         ii = r1+1
      lalt = len(boxL)

   boxL       = removeBoxContainedInOtherBox(boxL)
   boxL       = boxFilterByInterval(boxL, bx, by, progressbar=progressBar)
   
   for ii in range(r2):
      #boxL = removeBoxContainedInOtherBox(boxL, progressbar=progressBar)
      boxL = unionOfAllBoxes(boxL, progressbar=progressBar)

   boxL       = boxFilterByInterval(boxL, bx, by, progressbar=progressBar)
 
   return([boxL, boxL0])

################################################################################################

def checkRowCharacters(A, bx=[7,50], by=[7,50], r1=2, progressBar=False):

   boxL0      = getBoxFromContours(A, progressbar=False, hr=0)
   boxL       = boxFilterByInterval(boxL0, [0,bx[1]], [0,A.shape[0]], progressbar=progressBar)
   #boxL       = removeBoxContainedInOtherBox(boxL, progressbar=progressBar)

   for ii in range(r1):
      boxL = removeBoxContainedInOtherBox(boxL)
      boxL = unionOfAllBoxes(boxL, progressbar=progressBar)

   boxL       = removeBoxContainedInOtherBox(boxL)
   boxL       = boxFilterByInterval(boxL, [0    , A.shape[1]], [by[0], A.shape[0]], progressbar=progressBar)
   boxL       = boxFilterByInterval(boxL, [bx[0], A.shape[1]], [0    , A.shape[0]], progressbar=progressBar)

   return([boxL, boxL0])



################################################################################################

def skewnessOfWholePage(M2, bx=[8,50], by=[8,50], r1=2, r2=2, wsL = [ 20,30,40,50,60,70,80,90,100], startList = [ 0,5,15,20], stepSize=10, lb=20, lb2=20):

   M2T            = cv.threshold(M2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
   imgM2T         = Image.fromarray(M2T)
   imgM2TF        = imgM2T.filter(ImageFilter.CONTOUR())
   imgM2TF        = imgM2TF.filter(ImageFilter.EMBOSS())

   At             = np.array(imgM2TF)
   boxLAt, boxAt0 = checkPage(At, bx=bx, by=by, r1=r1, r2=r2, progressBar=True)
 
   a,A            = getAverageHeightOfTextLine(wsL, M2.copy(), boxLAt, startList)
   # checL len(A)>0??
   M, erg, m, cut, wcut = stepThroughImage(M2.copy(), boxLAt, int(A[0][0]), stepSize, start=0, lb=lb2)
   maxLetters     = max(list(map(lambda x: x[2], erg)))
   a,b,c          = list(filter( lambda x: x[2] ==maxLetters , erg))[0]
   W              = list(filter( lambda x: a <= x[1] <= x[3] <= b, boxLAt))
   W              = list(filter( lambda x: abs(x[2]- x[0])> lb , W))
   W.sort(key=lambda x: x[2])

   erg = []
   for ii in range(0, len(W)-2):
      x1,y1,x2,y2 = W[ii]
      if y2 <= W[ii+1][3] <= W[ii+2][3]:
         erg.append([W[ii], W[ii+1], W[ii+2]])
 
   alphaL = list(map(lambda x: getAlpha(x), erg))  
   degL   = list(map(lambda x: x[1], alphaL))

   return([W, boxLAt, boxAt0, cut, degL, imgM2TF, erg])

################################################################################################

def removeBoxesOfHorizonzalAndVerticalLines(boxLt, ax=[50,150], bx=[0,10]):

   boxL    = boxLt.copy()
   boxL_x  = boxFilterByInterval(boxL, ax, bx, progressbar=True)
   boxL_y  = boxFilterByInterval(boxL, bx, ax, progressbar=True)
   L       = unique(boxL_x + boxL_y)

   for ii in range(len(L)):
      box = L[ii]
      boxL.remove(box)

   return([boxL, L])

################################################################################################

class test:
   def __init__(self, M2, wsL, startList, a, b, xb, yb, angleList, name='test' ):
      self.name  = name   
      self.wsL   = wsL
      self.startList = startList
      self.a         = a
      self.b         = b
      self.xb        = xb
      self.yb        = yb
      self.M2        = M2
      self.angleList = angleList

################################################################################################

   def makeTestMatrix(self, angle):

      M2rot          = rotateImage(self.M2.copy(), angle)
      M2T            = cv.threshold(M2rot, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
      imgM2T         = Image.fromarray(M2T)
      imgM2TF        = imgM2T.filter(ImageFilter.CONTOUR())
      imgM2TF        = imgM2TF.filter(ImageFilter.EMBOSS())
      At             = np.array(imgM2TF)

      Bt             = At[self.a:self.b, :]
      M2t            = M2rot[self.a:self.b, :]
      boxL0          = getBoxFromContours(Bt, progressbar=False, hr=0)
      boxL1          = boxFilterByInterval(boxL0, self.xb, self.yb, progressbar=True)
 
      self.M2t       = M2t
      self.boxL1     = boxL1

      return([Bt, M2t, boxL1])

################################################################################################

   def testBox(self, Mt, boxL, windowSize, stepSize, start):

      erg  = []
      tt   = []
      zz   = 0

      for ii in range(start, Mt.shape[0]-windowSize, stepSize):
         W = list(filter( lambda x: ii <= x[1] <= x[3] <= ii+windowSize, boxL))
         erg.append([ii, ii+ windowSize, len(W), W])
         tt = tt+W
         zz = zz+1
          

      tt = unique(tt)
      r1 = len(tt)/len(boxL)
      r2 = (1-r1)/(2*zz)

      return([r1, r2, erg])

################################################################################################

   def makeTest(self, boxL1, withoutAngles=False):
    
      L     = []
      R     = tqdm(self.angleList)
      if withoutAngles:  
         R = [0]
      for angle in R:
         boxL1t  = transformBoxes(boxL1,angle)
         for jj in range(len(self.startList)):
            start = self.startList[jj]
            for ii in range(len(self.wsL)):
               w       = self.wsL[ii]   
               r1, r2, erg  = self.testBox(self.M2t, boxL1t, w, w, start=start)
               L.append([jj, ii, r1, r2, angle])

      return(L)

################################################################################################

   def evaluateData(self, ii,jj, angle, plotAllBoxes=False):

      w               = self.wsL[ii]
      start           = self.startList[jj]
      boxL1t          = transformBoxes(self.boxL1,angle)
      r1, r2,ergt     = self.testBox(self.M2t, boxL1t, w, w, start)
      erg             = list(map(lambda x: [x[0], x[1], x[2]], ergt))
      WL              = self.boxL1
      if not(plotAllBoxes):
         W               = list(map(lambda x: x[3], ergt))
         WL              = []
         for ii in range(len(W)):
            WL = WL + W[ii]
      
      fak            = (2*90)/pi
      M2rot          = rotateImage(self.M2.copy(), angle*fak  )

      M   = myShow(nn=0, L=[WL], A=self.M2t.copy(), progressbar=True)
      D   = ImageDraw.Draw(M)
      for ii in range(len(erg)):
         y1,y2,c = erg[ii]
         D.line( (0, y1, 3000, y1), fill='blue',   width=5)
         D.line( (0, y2, 3000, y2), fill='yellow', width=3)
   
      D.text( (10,10), "a="+str(self.a)+ " b=" + str(self.b) ,fill=(255,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))

      return([M, M2rot, boxL1t,  WL])

################################################################################################

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

def transformBoxes(boxL1, r):

   boxL1PC           = list(map(lambda x: [ cart2pol([x[0], x[1]]), cart2pol([x[2], x[3]])], boxL1))
   boxL1PCt          = list(map(lambda x: [ (x[0][0], x[0][1]-r), (x[1][0], x[1][1]-r)], boxL1PC))
   boxL1t            = list(map(lambda x: pol2cart(x[0])+ pol2cart(x[1]), boxL1PCt))

   return(boxL1t)

###################################################################################################################

def rotateImage(N, angle, center=(0,0)):

   Nflip = flipImage(N)
   M     = Image.fromarray(Nflip).rotate(angle=angle, center=center  )
   N1    = flipImage(np.array(M))

   return(N1)

###################################################################################################################

def drawSepLines(M, erg):

   img, A = matrixToRGBImage(M)
   D      = ImageDraw.Draw(img)
   for ii in range(len(erg)):
      y1,y2,c = erg[ii]
      D.line( (0, y1, 3000, y1), fill='blue',   width=5)
      D.line( (0, y2, 3000, y2), fill='yellow', width=3)
      D.text( (0, y2), "ii=" + str(ii) + "y2= " + str(y2) ,fill=(255,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15))   

   return(img)

###################################################################################################################

def getInter(d, Nt, boxL1t):

   LERG   = []
   N      = Nt.copy()

   for ii in range(N.shape[0]):
      l = list(filter( lambda x: x[1]<= ii <= x[3] , boxL1t))
      if len(l) <=d:
         LERG.append(ii)
         N[ii, :]=0

   return([N, LERG])

###################################################################################################################


def drawBoxList(img, boxL, color='red', den=False):
   draw    = ImageDraw.Draw(img)
   for ii in range(len(boxL)):
      x1,y1,x2,y2 = box = boxL[ii]
      draw.rectangle(box, width=3, outline=color)
      if den:
         draw.text( (x1,y1), str(ii) ,fill=(255,0,0), font=ImageFont.truetype(font='Roboto-Bold.ttf', size=15)) 

   return(img)

###################################################################################################################


def boxListContained(boxL1, boxL2): # gibt alle box in boxL2 aus, die auch in boxL1 sind.
   ERG = []
   for ii in range(len(boxL1)):
      x1,y1,x2,y2 = box = boxL1[ii]
      IS = getIntersection(box, boxL2, 0, 0)
      IS = list(filter( lambda x: x1 <= x[0] <= x[2] <= x2 and y1 <= x[1] <= x[3] <= y2 , IS)) 
      ERG.extend(IS)

   return(ERG)

###################################################################################################################


def boxListComplement(boxL1, boxL2):  # gibt alles Boxen von boxL1 zurück, die nichtin boxL2 sind
   ERG = []
   for ii in range(len(boxL1)):
      box = boxL1[ii]
      if not( box in boxL2):
         ERG.append(box)

   return(ERG)

###################################################################################################################

def makeBoxList(BOXLt):
   BOXL = []
   for ii in range(len(BOXLt)):
      boxt      = BOXLt[ii]
      xl, yl    = list(boxt[:,0]), list(boxt[:,1])
      box       = [ min(xl), min(yl), max(xl), max(yl)]
      BOXL.append(box)

   return(BOXL)

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

#################################################################################################
#################################################################################################
#################################################################################################

#***
#*** MAIN PART
#***
#
#  exec(open("OCRTextCutter.py").read())
#
##   

######################################### initial part ############################################################
   
#engine                = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
#con                   = engine.connect()
#train                 = dOM.JPGNPNGGenerator('/home/markus/anaconda3/python/pngs/train/'    ,  'train'     , '/home/markus/anaconda3/python/pngs/train/word/', 'train'     , 1, 0, False, 800, 'cv')        
#train.Q               = []
#train.engine          = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
#train.con             = engine.connect()

############################################  main  ###############################################################

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

#BOXL = MISC.loadIt(pathData+'BOXL-02.02.2022-20:54:42')


O0                 = getattr(BOXL, 'muenchen-trudering-bestandsverzichnis')
O1                 = getattr(BOXL, 'bonn-beuel-abteilung-1')
O2                 = getattr(BOXL, 'bonn-beuel-abteilung-2')
O3                 = getattr(BOXL, 'bonn-beuel-abteilung-3')
O4                 = getattr(BOXL, 'abteilung-1-schreibmaschineSchiefGescanntAberGutLesbar')
O5                 = getattr(BOXL, 'leipzig-leipzig-abteilung-1')
O6                 = getattr(BOXL, 'siegburg-obermenden-abteilung-1')
O7                 = getattr(BOXL, 'siegburg-obermenden-abteilung-3')
O8                 = getattr(BOXL, 'muenchen-muenchen-abteilung-3')
O10                = getattr(BOXL, 'goslar-bündheim-bestandsverzeichnis')
O11                = getattr(BOXL, 'goslar-bündheim-abteilung-2')




O          = getattr(BOXL, NL[0].split('.')[0]) 
O.preprocessing.frontierBlackWhite = 200
try:
   a = getattr(O, 'calcAll')
except:
   O.calcAll()

O.makeM2()


M2                = O.JPG.M2[:,:,0]
aL                = np.arange(-5, 5.5, 0.5)
piL               = aL*(pi/(2*90))
TM                = test(M2, wsL=[80, 90,100,110], startList=[0,10], a=0, b=3000, xb=[12,50], yb=[12,50], angleList=piL)
Bt, M2t, boxL1    = TM.makeTestMatrix(0)

#L1                = TM.makeTest(boxL1)  

#nn                = 3
#m1                = max(list(map(lambda x: x[nn], L1)))
#T1                = list(filter(lambda x: x[nn]==m1, L1))[0]
#jj1, ii1, r11, r12, angle1 = T1

fak               = (2*90)/pi
angle1            = -0.008726646259971648
M2rot             = rotateImage(M2t.copy(), angle1*fak  )
T1                = test(M2rot, wsL, startList, a, b, xb=[0,1000], yb=[0,1000], angleList = piL)
Btrot, M2t, boxL1 = T1.makeTestMatrix(0)


def pBL(M2, boxL):
   imgBIGL, _  = matrixToRGBImage(M2)
   imgBIGL     = drawBoxes(boxL = boxL, sizeMat = (3000,3000), img=imgBIGL, width=1, progressbar = True, makeDenotations=False)
   imgBIGL.show()


def getComplement(B, CBOX):
   BIGL0     = B.copy()
   zz       = 0
   R = tqdm(range(len(CBOX)))
   R.set_description('complement')

   for ii in R:
      cbox = CBOX[ii]
      erg  = getAllBoxesContainedInGivenBox(cbox, B)[0]
      for jj in range(len(erg)):
         try:
            BIGL0.remove(erg[jj])
         except:
            zz=zz+2

   return(BIGL0)



makeCraft         = True
image             = M2rot
output_dir        = 'outputs/'
craft             = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

if makeCraft:
   try: 
      d = len(prediction_result.keys())
   except:
      prediction_result = craft.detect_text(M2rot)
      craft.unload_craftnet_model()
      craft.unload_refinenet_model()

      img, A  = matrixToRGBImage(M2rot)
      draw    = ImageDraw.Draw(img)

      BOXLt   = prediction_result['boxes']
      CBOX    = makeBoxList(BOXLt)
      #BOXL    = list(map( lambda x: list(np.round(x)), BOXL ))
      #BOXL    = list(map( lambda x: list(map(lambda y: int(y), x)), BOXL ))

      imgMB             = drawBoxes(boxL = CBOX, sizeMat = (3000,3000), sizeFont=25,img=img, width=3, progressbar = True, makeDenotations=True)
      imgMB.show()

      BIGL0 = getComplement(boxL1, CBOX)


BIGL1_y = filterBoxes(BIGL0, [0,3000], [0,10])
BIGL1_x = filterBoxes(BIGL0, [0,10], [0,3000])

L = BIGL1_y + BIGL1_x
for ii in range(len(L)):
   box = x1,y1,x2,y2 = L[ii]
   M2rot[y1:y2, x1:x2] = 255

T1                = test(M2rot, wsL, startList, a, b, xb=[0,1000], yb=[0,1000], angleList = piL)
Btrot, M2t, boxL2 = T1.makeTestMatrix(0)

BIGL0   = getComplement(boxL2, CBOX)
BIGL1_y = filterBoxes(BIGL0, [0,3000], [0,10])
BIGL1_x = filterBoxes(BIGL0, [0,10], [0,3000])

L = BIGL1_y + BIGL1_x
for ii in range(len(L)):
   box = x1,y1,x2,y2 = L[ii]
   M2t[y1:y2, x1:x2] = 255

T1                = test(M2t, wsL, startList, a, b, xb=[0,1000], yb=[0,1000], angleList = piL)
Btrot, M2t, boxL3 = T1.makeTestMatrix(0)

BIGL0   = getComplement(boxL3, CBOX)
BIGL1_y = filterBoxes(BIGL0, [0,3000], [0,10])
BIGL1_x = filterBoxes(BIGL0, [0,10], [0,3000])

L = BIGL1_y + BIGL1_x
for ii in range(len(L)):
   box = x1,y1,x2,y2 = L[ii]
   M2t[y1:y2, x1:x2] = 255


T1                = test(M2t, wsL, startList, a, b, xb=[0,1000], yb=[0,1000], angleList = piL)
Btrot, M2t, boxL4 = T1.makeTestMatrix(0)

BIGL0   = getComplement(boxL4, CBOX)
BIGL1_y = filterBoxes(BIGL0, [0,3000], [0,10])
BIGL1_x = filterBoxes(BIGL0, [0,10], [0,3000])

L = BIGL1_y + BIGL1_x
for ii in range(len(L)):
   box = x1,y1,x2,y2 = L[ii]
   M2t[y1:y2, x1:x2] = 255

Image.fromarray(M2t).show()


"""
nn       = 20
imgE, A  = matrixToRGBImage(M2rot)
erg      = getAllBoxesContainedInGivenBox(CBOX[nn], boxL1, overlapping=True, overlappingFactor=0.2)
B        = erg[0]
B        = filterBoxes(erg[0], [0, 50], [5,100])
imgE     = drawBoxes(boxL = B + [CBOX[nn]], sizeMat = (3000,3000), img=imgE, width=3, progressbar = True, makeDenotations=False)
#imgE.show()
"""

ergL     = BIGL1_y + BIGL1_x
for ii in range(1,0):
   ergL = unionOfAllBoxes(ergL, progressbar= True, glx=0, gly=0)
   ergL = removeBoxContainedInOtherBox(ergL)

#ergL = unionOfAllBoxes(ergL, progressbar= True, glx=10, gly=10)
#ergL = removeBoxContainedInOtherBox(ergL)


#pBL(M2rot, ergL)

#imgBIGL, _  = matrixToRGBImage(M2rot)
#imgBIGL     = drawBoxes(boxL = ergL, sizeMat = (3000,3000), img=imgBIGL, width=3, progressbar = True, makeDenotations=False)
#imgBIGL.show()

"""

ergLn = []
cbox_x1,cbox_y1,cbox_x2,cbox_y2 = CBOX[nn]
for ii in range(len(ergL)):
   box = x1,y1,x2,y2 = ergL[ii]
   boxn = [ x1,cbox_y1, x2, cbox_y2]
   ergLn.append(boxn)

imgERG, _  = matrixToRGBImage(M2rot)
imgERG     = drawBoxes(boxL = ergLn + [CBOX[nn]], sizeMat = (3000,3000), img=imgERG, width=3, progressbar = True, makeDenotations=False)
imgERG.show()
"""

# boxListContained
# unionOfAllBoxes
# filterBoxes
# removeBoxContainedInOtherBox
#def boxListComplement(boxL1, boxL2):  # gibt alles Boxen von boxL1 zurück, die nichtin boxL2 sind
#def boxListContained(boxL1, boxL2): # gibt alle box in boxL2 aus, die auch in boxL1 sind.
#def unionOfAllBoxes(boxL, progressbar, glx=0, gly=0):
#def getIntersection(box1, boxL, glx, gly):
#def getAllBoxesContainedInGivenBox(box1, boxL, overlapping=False, overlappingFactor = 0.7):
#isContainedIn(box1, box2):
#unionOfBoxes(box1, box2):




#for ii in range(3):
#   boxL4 = unionOfAllBoxes(boxL4, progressbar= True, glx=0, gly=0)
#   boxL4 = filterBoxes(boxL4, [0, 100], [0,50])

   #boxL4 = removeBoxContainedInOtherBox(boxL4)
   #boxL4 = boxListContained(boxL4t, boxL4)



