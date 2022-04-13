
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
from bs4 import BeautifulSoup

import sys, subprocess
sys.path.append('/home/markus/anaconda3/python/modules')


#import misc_v9 as MISC
#import scatteringTransformationModule_2D_v9 as ST


pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real



class RBA:

   def __init__(self, pathToPDF, pathToPNG, what, pdfFilename):
      self.pathToPDF   = pathToPDF
      self.pathToPNG   = pathToPNG
      self.what        = what
      self.pdfFilename = pdfFilename
      
   ###########################################################################   
    
   def merge(self, Xt, mm=20):
      X  = Xt.copy()
      jj = 0
      while jj <= len(X)-2:
         x = X[jj]
         y = X[jj+1]
         #print('x: ' + str(x) + ' y: ' + str(y))
         if y[0]-x[1] <= mm:
            #print('x: ' + str(x) + ' y: ' + str(y))
            X[jj] = [x[0], y[1]]
            X.remove(y)
         else:
            jj = jj+1           
      return(X)    
   
   ###########################################################################
   
   def calcAvgHeightLetters(self, W):
      a  = list(map(f3, W))
      b  = list(map(f4, a))
      dd = list(set(b))
      ee = list(map(b.count, dd))
      n  = ee.index(np.array(ee).max())

      erg = dd[n]+0.1
      return(erg) 
   
   ###########################################################################
    
   def uInt(self, X):
      def el(l, w):
         erg = []
         for ii in range(0, len(l)):
            erg.append([l[ii],w])
         return(erg)

      def f1(x):
         return(x[1])

      def f0(x):
         return(x[0])

      A,B   = el(list(map(f0, X)), 'a'), el(list(map(f1, X)), 'b')
      S     = A+B; S.sort()
      H,G   = list(map(f0, S)), list(map(f1, S))
      IL,zz = [], 0
      a,w   = H[0], G[0]
      I     = [a]
      while zz < len(H):
         if G[zz] == 'a':
            if zz ==0:
               I  = [H[zz]]
            if zz>0 and G[zz-1] == 'b':
               I  = [H[zz]]
         if G[zz] == 'b':
            if zz == len(H)-1:
               I.append(H[zz])
               IL.append(I)
            else:
               if G[zz+1] == 'a':
                  I.append(H[zz])
                  IL.append(I)
         zz = zz+1
      
      return(IL) 
    
   ###########################################################################  
   
   def filterWords(self,uu,dd):
   
      X = []
   
      for w in self.words:
         if (float(w["ymin"]) >= uu) and (float(w["ymax"]) <= dd): 
            X.append([float(w["xmin"]), float(w["xmax"])])

      for b in self.blocks:
         if (float(b["ymin"]) >= uu) and (float(b["ymax"]) <= dd): 
            X.append([float(b["xmin"]), float(b["xmax"])])
        
      
      self.X = X
     
     
     
   ###########################################################################  
     
   def Uber(self, page, uu, dd):
      inputfname   = self.pdfFilename + ".pdf"
      outputfname  = self.pdfFilename + "-pdfToText-p" + str(page) + ".xml"
      ss           = "pdftotext -bbox-layout -f " + str(page) + " -l " + str(page) + " -htmlmeta " + self.pathToPDF + inputfname + " " + self.pathToPDF+outputfname; 
      #print(ss)
      subprocess.check_output(ss, shell=True,executable='/bin/bash')

      soup_pdfToText  = BeautifulSoup(open(self.pathToPDF + outputfname), "html.parser")
      width_ptt       = float(soup_pdfToText.find_all('page')[0]["width"].encode())
      self.words      = soup_pdfToText.find_all('word')
      self.blocks     = soup_pdfToText.find_all('block')
      X               = []
      Xt              = []
      self.X          = self.filterWords(uu,dd)
   
      if len(X)>0:
         self.Xt = self.uInt(self.X)
  
    
   
   ###########################################################################
   
   def padding(self, Ct, makeEven=True, plist=[0,0,0,0], cont=255):
      o,u,l,r         = plist
      n,m             = Ct.shape
      if makeEven:
         if (n + o + u)%2==1:
            o = o+1  
         if (m + l + r)%2==1:
            l = l+1
      C               = np.ones((n+o+u, m+l+r))*cont
      C[o:n+o, l:l+m] = Ct
      
      return(C) 
    
   ########################################################################### 
    
   def generateMatrixFromPNG(self, fname):
      return(self.padding(np.asarray(Image.open(fname).convert('L'))))
    
   ########################################################################### 
    
   def getFormatFromPDFPage(self, page):
      ss = "pdfinfo -f " + str(page) + " -l " + str(page) + " " + self.pathToPDF+ self.pdfFilename + ".pdf" + " | grep -i '" + str(page) + " size'"  
      tt = subprocess.check_output(ss, shell=True,executable='/bin/bash')
      tt = tt.decode('utf-8')
      tt = tt.replace(' ', '')
      aa = tt.split('x')
      x = float(aa[0].split(':')[1])
      y = float(aa[1].split('pts')[0]) 
      
      return([x,y])   
      
   ###########################################################################
      
   def HMTOC_ruleBased(self, page, mm1=10, mm2=18, uu=70, dd=750):
      [x,y]           = self.getFormatFromPDFPage(page)
      format = 'portrait'
      if x > y:
         format = 'landscape'
      filename        = self.pdfFilename + '-' + str(page) + '-'+ format +'-' + self.what + '.png'
      Xt, words       = self.Uber(page, uu,dd)
      erg             = 1
      X3,X2           = [],[]   
      if len(Xt)>0:
         X3              = self.merge(Xt, mm1)
         X2              = self.merge(Xt, mm2)
         if sum(1*(np.diff(X3)>150))[0] >2:
            erg = 3
         if sum(1*(np.diff(X2)>180))[0] >1:
            erg = 2
      
      self.erg = erg
      self.Xt  = Xt
      self.X2  = X2
      self.X3  = X3
      self.filename = filename
        

###########################################################################

def f1(x):
   return(float(x["xmin"]))

###########################################################################

def f11(x):
   return(float(x["xmax"]))

###########################################################################

def f2(x):
   return(float(x["ymin"]))

###########################################################################

def f3(x):
   return([float(x["ymin"]), float(x["ymax"])])

###########################################################################

def f4(x):
   return(abs(x[1]-x[0])) 

###########################################################################

def displayI(Xt, C, ypos=750):

   image          = Image.fromarray(C)
   draw           = ImageDraw.Draw(image)
   
   for xt in Xt:
      draw.line( ((xt[0], ypos), (xt[1], ypos)), width=5)

   image.show()
   
###########################################################################


   
#***
#*** MAIN PART
#***
#
#  exec(open("testWallBrickMethod-v3.py").read())
#
## 

"""
HMTOC_labels = np.loadtxt("/home/markus/anaconda3/python/data/AnnoHMTOC_train_hochkant_v2.csv", delimiter=',')
for ii in range(len(HMTOC_labels)):
   if HMTOC_labels[ii] == 0:
      HMTOC_labels[ii] = 1
HMTOC_labels = list(HMTOC_labels)


pathToPDF       = '/home/markus/anaconda3/python/pngs/train_hochkant/'
pathToPNG       = '/home/markus/anaconda3/python/pngs/train_hochkant/word/'
what            = 'word'
pdfFilename     = 'train_hochkant'
ERG             = []

for ii in range(0, 123):
   page            = ii+1
   ERG.append(HMTOC_ruleBased(page, what=what, pathToPNG=pathToPNG, pathToPDF=pathToPDF, pdfFilename=pdfFilename) )      

"""


