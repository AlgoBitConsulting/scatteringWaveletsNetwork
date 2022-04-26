import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import sys, subprocess

import DFTForSCN_v7 as DFT
import scatteringTransformationModule_2D_v9 as ST

import tkinter
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import pickle
from functools import partial
from joblib import Parallel, delayed
import multiprocessing as mp
import timeit
from datetime import datetime 
import pywt
from copy import deepcopy
from tqdm import tqdm

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin

matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag

#############################################################################

def f(i):
   def fi(x):
      return(x[i])
   return(fi)
 
f0 = f(0)
f1 = f(1)
f2 = f(2)
f3 = f(3)

#############################################################################  

class SWO_1D:
    
   def __init__(self, C, dO, rad):
      self.C     = C
      self.dO    = dO
      self.rad   = rad
      self.rows  = self.C.shape[0]
      self.cols  = self.C.shape[1] 

      self.dX    = round(self.cols/self.dO,2)
      self.dY    = round(self.rows/self.dO,2)
      self.OX    = np.arange(-self.dO/2, self.dO/2, self.dO/self.cols)
      self.X     = np.arange(-self.dX/2, self.dX/2, self.dX/self.cols) 
      self.OY    = np.arange(-self.dO/2, self.dO/2, self.dO/self.rows)
      self.Y     = np.arange(-self.dY/2, self.dY/2, self.dY/self.rows) 

#############################################################################  

class SWO:
    
   def __init__(self, C, dO1,dO2, rad, s1, nang, nu,sl):
      self.C     = C
      self.dO1   = dO1
      self.dO2   = dO2
      self.rad   = rad
      self.s1    = s1
      self.nang  = nang
      self.nu    = nu
      self.sl    = sl

      self.rows  = self.C.shape[0]
      self.cols  = self.C.shape[1] 
      self.dX    = round(self.cols/self.dO1,3)
      self.dY    = round(self.rows/self.dO2,3)
      self.Xdt   = round(self.dX/(self.cols-1),3)
      self.Ydt   = round(self.dY/(self.rows-1),3)
      self.O1dt  = round(self.dO1/(self.cols-1),3)
      self.O2dt  = round(self.dO2/(self.rows-1),3)
      if s1>0:
         self.alpha = sqrt( 2*log(1/self.s1)*(1/self.rad**2)) 
      else:
         self.alpha=1

      #self.O1    = np.arange(-self.dO1/2, self.dO1/2 + self.O1dt, self.O1dt)
      #self.O2    = np.arange(-self.dO2/2, self.dO2/2 + self.O2dt, self.O2dt)      
      #self.X     = np.arange(-self.dX/2, self.dX/2 + self.Xdt, self.Xdt)
      #self.Y     = np.arange(-self.dY/2, self.dY/2 + self.Ydt, self.Ydt)
     
      self.O1    = np.linspace(-self.dO1/2, self.dO1/2, self.cols)
      self.O2    = np.linspace(-self.dO2/2, self.dO2/2, self.rows)      
      self.X     = np.linspace(-self.dX/2, self.dX/2,   self.cols)
      self.Y     = np.linspace(-self.dY/2, self.dY/2,   self.rows)
     

      self.info  = '(cols, dO1, dX=cols/dO1, M); (rows, dO2, dY=rows/dO2, N)'
     

   def makeO1O2(self, lam):
      return(transformOFF(self.cols, self.rows, self.dO1, self.dO2, lam*self.alpha, self.rad))

#############################################################################  

class SWO_2D:
    
   def __init__(self, C, a, b, rad, s1, nang, nu,sl):
      self.C     = C
      self.rows  = self.C.shape[0]
      self.cols  = self.C.shape[1] 
      self.rad   = rad
      self.s1    = s1
      if s1>0:
         self.alpha = sqrt( 2*log(1/self.s1)*(1/self.rad**2)) 
      else:
         self.alpha=1
      
      self.nang  = nang
      self.nu    = nu
      self.sl    = sl

     
      self.a        = a
      self.b        = b
      self.X        = np.linspace(-self.a, self.a,   self.cols, endpoint=False)
      self.Y        = np.linspace(-self.b, self.b,   self.rows, endpoint=False)
      x1,y1         = np.meshgrid(self.Y, self.X, indexing='ij')
      pos           = np.empty(x1.shape + (2,))
      pos[:, :, 0]  = y1; pos[:, :, 1] = x1
      self.Z        = pos

      self.dx       = abs(self.X[1]-self.X[0])
      self.dy       = abs(self.Y[1]-self.Y[0])

      self.F_N      = 1/self.dx
      self.F_M      = 1/self.dy  
      self.O1       = np.linspace(-self.F_N/2, self.F_N/2,   self.cols, endpoint=False)
      self.O2       = np.linspace(-self.F_M/2, self.F_M/2,   self.rows, endpoint=False)
      w1,w2         = np.meshgrid(self.O2, self.O1, indexing='ij')
      pos           = np.empty(w1.shape + (2,))
      pos[:, :, 0]  = w2; pos[:, :, 1] = w1
      self.W        = pos
     
       
#############################################################################       
     
class OB:
   def __init__(self, description):     
      self.description = description

class DD:
   def __init__(self):
      self.description = 'contains all filenames for model data'
      self.pos         = ''

#############################################################################  

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


class PAGE:
   def __init__(self, path, pdfFilename, what, page, p, q, widthOfStripe, stepSize, pad, SWO):
 
      self.path          = path
      self.pdfFilename   = pdfFilename
      self.what          = what
      self.page          = page
      
      self.p             = p
      self.q             = q
      self.widthOfStripe = widthOfStripe
      self.stepSize      = stepSize
      self.pad           = pad
      self.SWO           = SWO
      


   def makefname(self):
      self.fname         = self.path + self.what  +'/'+ self.pdfFilename + '-' + str(self.page) + '-' + self.what + '.png'   

   def padding(self, Ct, plist=[50,50,50,50], cont=255):
      o,u,l,r         = plist
      n,m             = Ct.shape
      C               = np.ones((n+o+u, m+l+r))*cont
      C[o:n+o, l:l+m] = Ct
      
      return(C)


   def generateStripes(self, eOB='b', onlyV=False, onlyH=False):
      
      page       = int(self.fname.split('-')[1])
      C_org, C   = ST.generateMatrixFromPNG(self.fname, 0, 0, cropBorders=False)   
      n,m        = C.shape
      CH         = self.padding(C, plist=[int(self.pad/2),0,0,0])
      CV         = self.padding(C, plist=[0,0,int(self.pad/2),0])
      
      if eOB =='b':
         CH    = self.padding(C, plist=[self.pad,0,0,0])
         CV    = self.padding(C, plist=[0,0,self.pad,0])
      if eOB == 'a':
         CH    = self.padding(C, [0,self.pad,0,0])
         CV    = self.padding(C, [0,0,0,self.pad])
    
      r          = int( max( (CH.shape[0]-self.widthOfStripe)/self.stepSize, (CV.shape[1]-self.widthOfStripe)/self.stepSize)) 
      LV         = []
      LH         = []
     
      tt = tqdm(range(r))
      tt.set_description_str("generateStripes...")
     
      for ii in tt:
         a       = ii*self.stepSize
         b       = a + self.widthOfStripe
         
         if b <= CH.shape[0] and not(onlyV):
            B         = np.ones(CH.shape)*255
            B[a:b, :] = CH[a:b, :]
            B         = adaptMatrix(B, n,m)
            Bt        = maxC(B, 3, False)
            Bt        = adaptMatrix(Bt, self.SWO.rows, self.SWO.cols)
            LH.append([Bt, a, b,-1, self.page, eOB])
            
      
         if b <= CV.shape[1] and not(onlyH):
            B = np.ones(CV.shape)*255
            B[:, a:b] = CV[:, a:b]
            B         = adaptMatrix(B, n,m)
            Bt        = maxC(B, 3, False)
            Bt        = adaptMatrix(Bt, self.SWO.rows, self.SWO.cols)
            LV.append([Bt, a, b,-1, self.page, eOB])
     
      self.LV = LV
      self.LH = LH
      self.CH = CH
      self.CV = CV
      
      return([LV, LH, CH, CV, C])



   def calculateSWCALL(self,LV, LH, onlyV=False, onlyH=False):

      ERG     = []
      DL_H    = list(map(f0, LH))
      DL_V    = list(map(f0, LV))
      ERG_Hx  = []
      ERG_Vx  = []
      
      if not(onlyV):   
         ERG_Hx  = makeIt(DL_H, self.SWO)      
      if not(onlyH):
         ERG_Vx  = makeIt(DL_V, self.SWO)
      
      annoHx  = list(map(f3, LH))
      annoVx  = list(map(f3, LV))
      
      return([ERG_Hx, ERG_Vx, annoHx, annoVx])
    
    
   def displayStripes(self, W='H', rf=None):

      global window
      global ii
      global L

      def key(event):
         global window
         global ii
         global L

         if event.char=='q': 
            ii = len(L)+1   
    
         if event.char=='j':
            ii = ii+5 
    
         window.quit()
         window.destroy()  


      if W=='V':
         L = self.LV
      else:
         L = self.LH
      
      #aaL = []
      #n,m = self.C.shape
      ii  = 0
      while ii < len(L):   
         if W=='H':
            C = self.CH.copy()     
         if W =='V':
            C = self.CV.copy()
         l              = L[ii]
         a,b            = l[1], l[2]
         if W=='H' and b <= C.shape[0]:
            C[a:b, : ] = C[a:b, :]*0.5 
         if W=='V' and b <= C.shape[1]:
            C[ :, a:b] = C[ :, a:b]*0.5 
         window         = tkinter.Tk()
         window.title("stripe nr " + str(ii) + ' a=' + str(a) + ' b=' + str(b)) 
         tkimg          = [None]
        
         if rf==None:
            tt             = 'annotation= '+str(l[3])
         else:
            tt = 'None'
         image          = Image.fromarray(C)
         draw           = ImageDraw.Draw(image)
         draw.text( (20,0), tt, font=ImageFont.truetype('Roboto-Bold.ttf', size=45))
         canvas         = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
         canvas.pack()
         image_tk       = ImageTk.PhotoImage(image)
         canvas.create_image( 0, 0, image=image_tk, anchor="nw" ,tags=("img")) 
         canvas.bind_all('<Key>', key)
   
         tkinter.mainloop()
         ii = ii+1
    
#############################################################################  

class COORDINATES:      
   def __init__(self, coord):
      self.x1, self.y1  = coord[0]
      self.x2, self.y2  = coord[1]
      self.coord        = coord
      
#############################################################################       
         
class BOX_STRIPES_COORD:  
   def __init__(self, coord, page):
      self.XY           = COORDINATES(coord)
      self.cord         = coord
      self.page         = page
      self.fname        = 'pngs/train_hochkant/block/train_hochkant-' + str(page) + '-block.png'
      self.stripes      = self.makeStripesCoord()
      
      
   def makeStripesCoord(self):
      C_org, C                         = ST.generateMatrixFromPNG(self.fname, 0, 0, cropBorders=False)        
      H                                = np.ones(C.shape)*255
      H[self.XY.y1+1:self.XY.y2-1, :]  = C[self.XY.y1+1:self.XY.y2-1, :] 
      self.H                           = H
      V                                = np.ones(C.shape)*255
      
      V[:, self.XY.x1+1:self.XY.x2-1 ] = C[:, self.XY.x1+1:self.XY.x2-1] 
      self.V                           = V
      self.C                           = C
      self.horizontalYCoord            = str(self.XY.y1+1) + ':' + str(self.XY.y2-1)
      self.verticalXCoord              = str(self.XY.x1+1) + ':' + str(self.XY.x2-1)

############################################################################# 

class BOX_STRIPES:
   def __init__(self, listOfCoord): #, p, widthOfStripe, stepSize, padding, eOB='b', q=0 ):
      cl                 = listOfCoord
      page               = cl[0][1]
      self.page          = page
      self.n             = len(listOfCoord)
      
      
      for ii in range(len(listOfCoord)):
         coord = listOfCoord[ii]
         name  = "box" + str(ii) 
         setattr(self, name, BOX_STRIPES_COORD(coord[0], page))   
       
      self.C = getattr(self, name, 'C') 
      self.C = self.C.C
      A      = getattr(self, name, 'C') 
      delattr(A, 'C')  
      
      aL_V = []
      bL_V = []
      aL_H = []
      bL_H = []
      
      for ii in range(0, self.n):
         box = getattr(self, 'box' + str(ii))
         aL_H.append(box.XY.y1)
         bL_H.append(box.XY.y2)
         aL_V.append(box.XY.x1)
         bL_V.append(box.XY.x2)
      
      self.aL_V = aL_V
      self.aL_H = aL_H
      self.bL_V = bL_V
      self.bL_H = bL_H
      
   #############################################################################     
    
   def annotationOfMovingStripe(self, a,b, pad, p_AL, q_AL, p_BL, q_BL, W, eOB='m'):  # W = H,V  
      """
      if W=='H':
         aL  = self.aL_H
         bL  = self.bL_H 
         if eOB=='b':
            aL  = list(np.array(self.aL_H)+ pad)
            bL  = list(np.array(self.bL_H)+ pad)  
         if eOB=='m':
            aL  = list(np.array(self.aL_H)+ int(pad/2))
            bL  = list(np.array(self.bL_H)+ int(pad/2))      
      if W=='V':
         aL  = self.aL_V
         bL  = self.bL_V
         if eOB =='b': 
            aL  = list(np.array(self.aL_V)+ pad)
            bL  = list(np.array(self.bL_V)+ pad)   
         if eOB=='m':
            aL  = list(np.array(self.aL_V) - int(pad/2))
            bL  = list(np.array(self.bL_V) - int(pad/2))  
      """
      
   
      aL  = self.aL_H
      bL  = self.bL_H
      if W=='V':
         aL  = self.aL_V
         bL  = self.bL_V
            
      erg = 0
      
      """
      if eOB=='b':
         A = sum( 1*( b-np.array(aL) <= p) *  1*( b-np.array(aL) >= q ) )
         B = sum( 1*( b-np.array(bL) <= p) *  1*( b-np.array(bL) >= q ) )       
         
      if eOB=='a':
         A = sum( 1*( np.array(aL)-a <= p) *  1*( np.array(aL)-a >= q ) )
         B = sum( 1*( np.array(bL)-a <= p) *  1*( np.array(bL)-a >= q ) ) 
      """
      
         
      if eOB=='m':
         m  = a + (b-a)/2 - 1*int(pad/2)
         la = m-np.array(aL)
         lb = m-np.array(bL)
         A = sum((1*(q_AL <= la) )*(1*( la <= p_AL)  ))
         B = sum((1*(q_BL <= lb) )*(1*( lb <= p_BL)  ))


         #A = sum( abs( m-np.array(aL)) <= p) 
         #B = sum( abs( m-np.array(bL)) <= p)  
         
      if B>0:
         erg = 2
         
      if A>0:
         erg = 3
         
      #if A>0 and B>0: 
      #   erg = 4
      
      return(erg)

   #############################################################################  
   
   def padding(self, Ct, plist=[50,50,50,50], cont=255):
   
      #print(str(plist))
      o,u,l,r         = plist
      n,m             = Ct.shape
      C               = np.ones((n+o+u, m+l+r))*cont
      C[o:n+o, l:l+m] = Ct
      
      return(C)
   
   #############################################################################  

   def calculateAnnotationsOfStripes(self,p_AL, q_AL, p_BL, q_BL, widthOfStripe, stepSize, pad, SWO, eOB='b', noV=False, noH=False):
     
      page       = self.page
      fname      = 'pngs/train_hochkant/block/train_hochkant-' + str(page) + '-block.png'
      C_org, C   = ST.generateMatrixFromPNG(fname, 0, 0, cropBorders=False)   
      n,m        = C.shape
      
      self.CH    = self.padding(C, plist=[int(pad/2),int(pad/2),0,0])
      self.CV    = self.padding(C, plist=[0,0,int(pad/2),int(pad/2)])
      
      if eOB =='b':
         self.CH    = self.padding(C, plist=[pad,0,0,0])
         self.CV    = self.padding(C, plist=[0,0,pad,0])
      if eOB == 'a':
         self.CH    = self.padding(C, [0,pad,0,0])
         self.CV    = self.padding(C, [0,0,0,pad])
      
      r          = int( max( (self.CH.shape[0]-widthOfStripe)/stepSize, (self.CV.shape[1]-widthOfStripe)/stepSize))
      self.r     = r
      self.LV    = []
      self.LH    = []
     
      for ii in range(r):
         a       = ii*stepSize
         b       = a + widthOfStripe
         
         if b <= self.CH.shape[0] and not(noH):
            B         = np.ones(self.CH.shape)*255
            B[a:b, :] = self.CH[a:b, :]
            B         = adaptMatrix(B, n,m)
            Bt        = maxC(B, 3, False)
            Bt        = adaptMatrix(Bt, SWO.rows, SWO.cols)
            self.LH.append([Bt,a,b,self.annotationOfMovingStripe(a, b, pad, p_AL, q_AL, p_BL, q_BL,  'H', eOB), page, eOB])
            
      
         if b <= self.CV.shape[1] and not(noV):
            B = np.ones(self.CV.shape)*255
            B[:, a:b] = self.CV[:, a:b]
            B         = adaptMatrix(B, n,m)
            Bt        = maxC(B, 3, False)
            Bt        = adaptMatrix(Bt, SWO.rows, SWO.cols)
            self.LV.append([Bt,a,b,self.annotationOfMovingStripe(a, b, pad, p_AL, q_AL, p_BL, q_BL, 'V', eOB), page, eOB]) 

   ############################################################################# 
   """
   def calculateAnnotationsOfCross(self,cl,p,h,l, SWO):
     
      page          = self.page
      fname         = 'pngs/train_hochkant/block/train_hochkant-' + str(page) + '-block.png'
      C_org, C      = ST.generateMatrixFromPNG(fname, 0, 0, cropBorders=False)   
      n,m           = C.shape
      stepSize      = h
      widthOfStripe = 2*l + h 
      Ct            = self.padding(C, plist=[l,l,l,l])
      r             = int( Ct.shape[1]/stepSize)
      CROSS_L       = []
      x             = l+ int(h/2)
      y             = l+ int(h/2)  
      ok            = True
      plo           = cl[0][0][0]
      pru           = cl[0][0][1]
      t             = int(h/2)
      s             = l + t
      
      while ok==True:
         D = np.ones(Ct.shape)*255
         D[y-s:y+s, x-t:x+t] = Ct[y-s:y+s, x-t:x+t]
         D[y-t:y+t: x-s:x+s] = Ct[y-t:y+t: x-s:x+s]  
         
         x       =  x + l
         if x > m-l:
            x = l + int(h/2)
            y = y + l
            if y > n-l:
               ok = False
       
   """        
         
   #############################################################################  

   def calculateSWCALL(self,LV, LH, SWO):

      ERG     = []
      DL_H    = list(map(f0, LH))
      DL_V    = list(map(f0, LV))
    
      ERG_Hx  = makeIt(DL_H, SWO)      
      ERG_Vx  = makeIt(DL_V, SWO)
      annoHx  = list(map(f3, LH))
      annoVx  = list(map(f3, LV))
      
      return([ERG_Hx, ERG_Vx, annoHx, annoVx])
    
     
   #############################################################################  

   def displayStripes(self, W='H', eOB='b'):

      global window
      global ii
      global L

      def key(event):
         global window
         global ii
         global L

         if event.char=='q': 
            ii = len(L)+1   
    
         if event.char=='j':
            ii = ii+5 
    
         window.quit()
         window.destroy()  


      if W=='V':
         L = self.LV
      else:
         L = self.LH
      
      #aaL = []
      n,m = self.C.shape
      ii  = 0
      while ii < len(L):   
         if W=='H':
            C = self.CH.copy()     
         if W =='V':
            C = self.CV.copy()
         l              = L[ii]
         a,b            = l[1], l[2]
         if W=='H' and b <= C.shape[0]:
            C[a:b, : ] = C[a:b, :]*0.5 
         if W=='V' and b <= C.shape[1]:
            C[ :, a:b] = C[ :, a:b]*0.5 
         window         = tkinter.Tk()
         window.title("stripe nr " + str(ii) + ' a=' + str(a) + ' b=' + str(b)) 
         tkimg          = [None]
         tt             = 'annotation= '+str(l[3])
         image          = Image.fromarray(C)
         draw           = ImageDraw.Draw(image)
         draw.text( (20,0), tt, font=ImageFont.truetype('Roboto-Bold.ttf', size=45))
         canvas         = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
         canvas.pack()
         image_tk       = ImageTk.PhotoImage(image)
         canvas.create_image( 0, 0, image=image_tk, anchor="nw" ,tags=("img")) 
         canvas.bind_all('<Key>', key)
   
         tkinter.mainloop()
         ii = ii+1


#############################################################################  

def usePywt(C, wvname, w, st, en):

   coeffs    = pywt.wavedec2(C, wvname)
   l         = len(coeffs)
   nn        = 0
   c         = [1*coeffs[0]]
     
   for ii in range(1,len(coeffs)):
      b  = np.zeros( (coeffs[ii][0].shape))
      nn = 0
      if ii>= st and ii <= en:
         nn = 1
      c.append([w[0]*nn*coeffs[ii][0], w[1]*nn*coeffs[ii][1], w[2]*nn*coeffs[ii][2]])

   erg = pywt.waverec2(c, wvname)
   return([erg, c, coeffs])        

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

def maxC(Cn,p,max=True):
  
   if p ==0:
      return(Cn)
      
   C     = Cn.copy()
   D     = C
   (m,n) = C.shape

   Ct     = padding(C)
   Ct     = padding(C, dirList=[2*(-(m/2)%p),2*(-(n/2)%p)], even=True, cont=0)
   r      = 2**p 
   
   (l,k)  = int(m/r), int(n/r)
   D      = np.zeros( (l, k))
   
   for ii in range(l):
      for jj in range(k):
         A        = C[r*ii:r*(ii+1), r*jj:r*(jj+1)]
         if max: 
            D[ii,jj] = min(255,A.max())
         else:
            D[ii,jj] = A.min()
   return(D) 

#############################################################################  

def adaptMatrix(M, mn,nn=-1):

   if nn==-1:
      t = deepcopy(mn)
      nn = t[1]
      mn = t[0]
      
   C   = M.copy()
   m,n = C.shape
   if m != mn:
      if m < mn:
         C=padding(C, dirList=[(mn-m)/2,0], even=True, cont=255)
      else:
         C = C[0:mn, :]
   if n != nn:
      if n < nn:
         C=padding(C, dirList=[0,(nn-n)/2], even=True, cont=255)
      else:
         C =C[:, 0:nn]                 
   return(C)  

#############################################################################  

def makeEven(x):
   return( x + x%2)     
     
#############################################################################  

def padding(C, dirList=[0,0], even=True, cont=255):

   l = C.shape
   if dirList == [0,0]:
      dirList = list(np.zeros(len(l)))
   if even==True:
      ln = list(map(makeEven, l))
   lp      = np.array(ln) + np.array(dirList)
   
   # is necessary adding one columns and/or row with zeros (=contd-value) and obtain even matrix D 
   e1      = np.array(ln)-np.array(l) 
   M       = np.zeros( (len(e1), 2), dtype=int )
   M[:, 1] = e1
   Z       = []
   for ii in range(M.shape[0]):
      Z.append(M[ii, :])
   Z = list(map(list, Z))
   D = np.pad(C, Z, 'constant', constant_values=(cont))   
   
   # is wished (i.e. dirList <> [0,0]) then matrix D is padded with "zeros" (=contd-values)
   
   M       = np.zeros( (C.ndim, 2), dtype=int )
   M[:, 0] = np.array(dirList)
   M[:, 1] = np.array(dirList)
   Z       = []
   for ii in range(M.shape[0]):
      Z.append(M[ii, :])
   Z = list(map(list, Z))
   E = np.pad(D, Z, 'constant', constant_values=(cont))  
   
   return(E)

#############################################################################  

def transformOFF(cols, rows, xB, yB, lam, rad):
   x             = np.linspace(-xB/2, xB/2, cols, endpoint=True)
   y             = np.linspace(-yB/2, yB/2, rows, endpoint=True)
   x1,y1         = np.meshgrid(y*lam, x*lam, indexing='ij')
   pos           = np.empty(x1.shape + (2,))
   pos[:, :, 0]  = y1; pos[:, :, 1] = x1
      
   return(pos)

#############################################################################  

def myPlot(x1,y1,Z1, col='blue', tit='NO TITLE', contour=False, withLabel=True):

   X1, Y1 = np.meshgrid(x1,y1)
   fig    = plt.figure()
   if contour==True:
      fig, ax = plt.subplots()
      CS = ax.contour(X1, Y1, Z1, linewidths=0.5)
      if withLabel:
         ax.clabel(CS, inline=1, fontsize=5)
      #fig.colorbar(CS, ax=ax[0])
      #plt.contour(X1,Y1,Z1, 20, cmap='RdGy')
      # plt.scatter(X1,Y1)
      plt.title(tit)
   else:
      ax     = plt.axes(projection='3d')
      ax.plot_wireframe(X1, Y1,Z1, color=col)
      ax.set_title(tit)

   return(fig)

#############################################################################  

def makeArray(fname,t=4, n=100):
   
   m         = int(n*0.71)
   img2      = Image.open(fname).convert('L')
   C1        = np.asarray(img2)
   rows,cols = C1.shape
   img3      = img2.resize((m,n),t)
   if rows < cols:
      img3    = img2.resize((n,m),t)
   C3        = np.flipud(np.asarray(img3))
   #C3        = np.array( C2==255,dtype='uint8')*255
   
   #print("n:" + str(n))

   return(C3)

#############################################################################  

def Mprint(M):
   print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in M]))

#############################################################################  

def makeM(M, rl, cl, r,c):
   b  = []
   for ii in range(M.shape[1] + len(cl)):
      b.append(r)
   N = []
   row = 0
   col = 0
   while row < M.shape[0]:
      a = []
      while col < M.shape[1]:
         if (col> 0 or row==0):
            a.append(str(M[row,col]))
         else:
            a.append(' ')
         if col in cl:
            for ii in range(cl.count(col)):
               a.append(c)
         col = col +1
      col = 0
      N.append(a)
      if row in rl:
         for ii in range(rl.count(row)):
            N.append(b)
      row = row+1
   return(N)

#############################################################################  

def pdfToBlackBlocksAndLines(filename,path,page, what, withSave=True, useXML = True):   

### what in ('block', 'line', 'word')

   def f4(x):
      return(list(map(float, x)))

   def f5(x):
      return([x["xmin"], x["ymax"], x["xmax"], x["ymin"]])

   def tr_pdf2txt_png(C, alpha, beta):
      x1 = alpha*C[0]
      x2 = alpha*C[2]
      y1 = beta*(pdfH-C[1])
      y2 = beta*(pdfH-C[3])
      return([x1,y1,x2,y2])

   def tr_pdf2txt_png(C, alpha, beta):
      x1 = alpha*C[0]
      x2 = alpha*C[2]
      y1 = beta*(pdfH-C[1])
      y2 = beta*(pdfH-C[3])
      return([x1,y1,x2,y2])

   def tr_pdftotext_png(C, alpha, beta):  
      x1 = alpha*C[0]
      x2 = alpha*C[2]
      y1 = beta*C[1]
      y2 = beta*C[3]
      return([x1, y1, x2, y2])

   
   inputfname      = filename + ".pdf"
   outputfname     = filename + "-pdfToText-p" + str(page) + ".xml"
   
   
   if not(useXML):   
      ss              = "pdftotext -bbox-layout -f " + str(page) + " -l " + str(page) + " -htmlmeta " + path + inputfname + " " + path+outputfname; 
      subprocess.check_output(ss, shell=True,executable='/bin/bash')

   soup_pdfToText  = BeautifulSoup(open(path + outputfname), "html.parser")
   whatList        = soup_pdfToText.find_all(what)
   #lines           = soup_pdfToText.find_all('word')
   pdfW            = float(soup_pdfToText.find_all('page')[0]["width"].encode())
   pdfH            = float(soup_pdfToText.find_all('page')[0]["height"].encode())

   img   = Image.new(mode="RGB",size=(round(pdfW), round(pdfH)), color=(255,255,255))
   draw  = ImageDraw.Draw(img)
   pngW  = img.size[0]
   pngH  = img.size[1]
   alpha = pngW/pdfW
   beta  = pngH/pdfH

   a     = list(map(f5, whatList))
   c     = list(map(f4, a))

   for x in c:
      draw.rectangle(tr_pdftotext_png(x, alpha, beta), fill=(0,0,0)) #, outline="#80FF00") # grün
   
   if withSave:
      pngfn = filename + "-" + str(page) + "-" +what + ".png"
      img.save(path+pngfn)

   return(img)

#############################################################################  

def gaussPDF(mu, sigma):
   def p(x):
      kk = sqrt(2*pi)*sigma
      xx = np.array(x)
      return((1/kk)*exp(- (xx-mu)**2/(2*sigma**2)))
   return(p)

#############################################################################  

def Wavelet_Morlet_1D(x, a,b,eta,sigma):
   #print('a=' +str(a)+ ' b=' + str(b) + ' eta=' + str(eta) + ' sigma=' + str(sigma))
   p                = gaussPDF(b, sigma)
   p1               = exp( 1j*2*pi*eta*( np.array(x)-b)*a)  
   erg2             = p1*np.array(p(x*a))
  
   return(erg2)

#############################################################################  

def make1D_psi_coef(a,b,eta,sigma,SWO, what='x'):
   C = SWO.C
   z = C.mean(1)
   k = SWO.dY
   m = Wavelet_Morlet_1D(SWO.Y, a,b,eta, sigma)
   if what=='x':
      z  = C.mean(0)
      k  = SWO.dX 
      m  = Wavelet_Morlet_1D(SWO.X, a,b,eta, sigma)

   erg1 = abs(DFT.appInvCFWithDFT_1D( DFT.appCFWithDFT_1D(z, SWO.dO)*DFT.appCFWithDFT_1D(m, SWO.dO), k ) )
   
   m  = Wavelet_Morlet_1D(SWO.Y, 2,0,0, 1)
   if what == 'x':
      m  = Wavelet_Morlet_1D(SWO.X, 2,0,0, 1)

   erg2 = abs(DFT.appInvCFWithDFT_1D( DFT.appCFWithDFT_1D(erg1, SWO.dO)*DFT.appCFWithDFT_1D(m, SWO.dO), k ) )

   return(round(sum(erg2),4))

#############################################################################  

def displaySlices(L):
   n,m,a,b = L.shape
   plt.figure(1)
   zz = 1
   for ii in range(n):
      for jj in range(m):
         plt.subplot(n, m, zz)
         img = Image.fromarray(L[ii,jj])
         plt.imshow(img)
         zz = zz+1   
   return(plt)

