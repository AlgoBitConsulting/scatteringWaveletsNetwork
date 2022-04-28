

### standardmäßig in python installiert
import sys, subprocess
from os import system
import os
import os.path
from PIL import Image, ImageDraw
from copy import deepcopy
import csv
from datetime import datetime 
import pickle
from functools import partial


### eigene Module
sys.path.append('/home/markus/python/scatteringWaveletsNetworks/modules')
sys.path.append('/home/markus/anaconda3/python/development/modules')

import scatteringTransformationModule as ST
import misc as MISC



### zu installierende Module
from tqdm import tqdm
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pytesseract
from pytesseract import Output
import pdf2image
from pdf2image import convert_from_path
from wand.image import Image as Wimage
import cv2 as cv



pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real




class matrixGenerator:

   #############################################################################  
   ###  padding                | adaptMatrix       | generateOnlyMatrices   ####  
   ###  maxC                   | generateSWC       | compressMatrix         ####  
   ###  generateMatrixFromPNG  | generateOnlySWC   | getLabels              ####
   ###  downSampling           | generateMatrices  |                        ####  
   #############################################################################


   def __init__(self, compressMethod):
      self.compressMethod = compressMethod
      self.errors         = ""
      self.warnings       = ""
   
   #############################################################################   
      
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

   #############################################################################

   def maxC(self, Cn, p, max=True):
      if p ==0:
         return(Cn)
      
      C     = np.matrix( Cn.copy(), dtype='uint8')
      D     = C
      (m,n) = C.shape
      r     = 2**p 

      Ct     = self.padding(C, False, [0,r-n%r ,0,r-m%r] )      
      (m,n)  = Ct.shape
      
      (l,k)  = int(m/r), int(n/r)
      D      = np.zeros( (l, k))
   
      for ii in range(l):
         for jj in range(k):
            A        = Ct[r*ii:r*(ii+1), r*jj:r*(jj+1)]
            if max: 
               D[ii,jj] = min(255,A.max())
            else:
               D[ii,jj] = A.min()
               
      D = self.padding(D)   
            
      return(D) 

   #############################################################################

   def generateMatrixFromImage(self, fname):
      return(self.padding(np.asarray(Image.open(fname).convert('L'))))

   #############################################################################
   
   #def compressMatrix(self, C): 
   #   C1 = self.downSampling(C, self.level)
   #   C2 = self.adaptMatrix(C1, self.mm, self.nn)
   #   
   #   return(C2)
      
   #############################################################################

   def downSampling(self, C, level, padding=True):
      Ct = C.copy()
      if level >= 1:
         zz = 0
         while zz < level:
            Ct = Ct[::2, ::2]
            zz = zz+1
         if padding:   
            Ct    = self.padding(np.asarray(Ct))
         
      return(Ct)

   #############################################################################

   def adaptMatrix(self, M, mn,nn=-1):
      if nn==-1:
         t = deepcopy(mn)
         nn = t[1]
         mn = t[0]
      
      C   = M.copy()
      m,n = C.shape
      if m != mn:
         if m < mn:
            C=self.padding(C,  makeEven=True, plist=[int((mn-m)/2),int((mn-m)/2),0,0], cont=255)
         else:
            C = C[0:mn, :]
      if n != nn:
         if n < nn:
            C=self.padding(C,  makeEven=True, plist=[0,0, int((nn-n)/2),int((nn-n)/2) ], cont=255)
         else:
            C =C[:, 0:nn]                 
      return(C)  

   #############################################################################

   def generateSWC(self, SWO_2D, con, SQL, downSamplingRate, adaptMatrixCoef ):
      
      DL             = self.generateMatrices(con, SQL, downSamplingRate, adaptMatrixCoef)
      CL             = list(map(f0, DL))
      HL             = list(map(f1, DL))
      PL             = list(map(f2, DL))
      PL             = list(map(lambda x: self.description + '-'+str(x), PL))
      
      AL             = makeIt(CL, SWO_2D, self.description )

      return([AL, HL, PL])

   #############################################################################

   def generateOnlySWC(self, SWO_2D, DL):
      
      CL             = list(map(f0, DL))
      HL             = list(map(f1, DL))
      PL             = list(map(f2, DL))
      PL             = list(map(lambda x: self.description + '-'+str(x), PL))
      
      AL             = makeIt(CL, SWO_2D, self.description )

      return([AL, HL, PL])

   #############################################################################

   def generateMatrices(self, con, SQL, downSamplingRate, adaptMatrixCoef):
      rs             = con.execute(SQL)
      colnames       = list(rs.keys())
      PNG_nn         = colnames.index('filenamePNG')
      hash_nn        = colnames.index('hashValuePNGFile')
      page_nn        = colnames.index('page')
      CL             = []
      
      R              = tqdm(list(rs))
      R.set_description('calculation of matrices for ' + self.description + ' ...')

      for row in R:
         r        = list(row)
         hash     = r[hash_nn]
         if hash != None:
            C1       = self.generateMatrixFromPNG(r[PNG_nn])
            C2       = self.downSampling(C1 , downSamplingRate)
            C3        = self.adaptMatrix(C2 , adaptMatrixCoef[0], adaptMatrixCoef[1])
            CL.append([C3, hash, r[page_nn]])
         
      return(CL)

   #############################################################################

   def generateOnlyMatrices(self, con, SQL):
      rs             = con.execute(SQL)
      colnames       = list(rs.keys())
      PNG_nn         = colnames.index('filenamePNG')
      hash_nn        = colnames.index('hashValuePNGFile')
      page_nn        = colnames.index('page')
      CL             = []
      
      R              = tqdm(list(rs))
      R.set_description('calculation of matrices for ' + self.description + ' ...')

      for row in R:
         r        = list(row)
         hash     = r[hash_nn]
         if hash != None:
            C       = self.generateMatrixFromPNG(r[PNG_nn])
            CL.append([C, hash, r[page_nn]])
         
      return(CL)

   #############################################################################

   def compressMatrix(self, CL, method):
      CL_comp = []
      R       = tqdm(range(len(CL)))
      R.set_description('compressing matrices...')
      for ii in R:
         C, hashValue, page = CL[ii]
         CL_comp.append([method(C), hashValue, page] )
         
      return(CL_comp)   

   #############################################################################

   def getLabels(self, SWO_2D, con, SQL, labelName ):
      rs             = con.execute(SQL)
      colnames       = list(rs.keys())
      label_nn       = colnames.index(labelName)
      hash_nn        = colnames.index('hashValuePNGFile')
      LL             = []
      HL             = []
      R              = tqdm(list(rs))
      R.set_description('calculation of labels for ' + self.description + ' ...')
      for row in R:
         r        = list(row)
         if r[hash_nn] != None:
            LL.append(r[label_nn])
            HL.append(r[hash_nn])
           
      return([LL, HL])

   ###########################################################################

   def printMat(A):
      print('\n'.join([''.join(['|{:5}'.format(item) for item in row]) for row in A]))

###########################################################################
          
  
















    
class imageOperations:

   def __init__(self, windowSize, stepSize, bound, part, ub):
      self.name       = "imageOperations"
      self.errors     = ""
      self.warning    = ""
      self.windowSize = windowSize
      self.stepSize   = stepSize
      self.bound      = bound
      self.part       = part      
      self.ub         = ub
      self.white      = 255
      self.colRatio   = [0.8, 1.2]
      self.totalRatio = 0.8

   ###########################################################################

   def pdfToBlackBlocksAndLines(self, pathPDFFilename, PDFfilename, pathOutput, page, what, format, withSave=True, useXML = True):   

   ### what in ('block', 'line', 'word')

      def f4(x):
         return(list(map(float, x)))

      def f5(x):
         return([x["xmin"], x["ymin"], x["xmax"], x["ymax"]])

      def tr_pdftotext_png(C, alpha, beta):  
         x1 = alpha*C[0]
         x2 = alpha*C[2]
         y1 = beta*C[1]
         y2 = beta*C[3]
         return([x1, y1, x2, y2])

   
      inputfname      = PDFfilename + ".pdf"
      outputfname     = PDFfilename + "-pdfToText-p" + str(page) + ".xml"
   
      if not(useXML):   
         ss              = "pdftotext -q -bbox-layout -f " + str(page) + " -l " + str(page) + " -htmlmeta " + pathPDFFilename + inputfname + " " + pathOutput+outputfname; 
         subprocess.check_output(ss, shell=True,executable='/bin/bash')

      soup_pdfToText  = BeautifulSoup(open(pathOutput + outputfname), "html.parser")
      whatList        = soup_pdfToText.find_all(what)
      pdfW            = float(soup_pdfToText.find_all('page')[0]["width"].encode())
      pdfH            = float(soup_pdfToText.find_all('page')[0]["height"].encode())

      img   = Image.new(mode="RGB",size=(round(pdfW), round(pdfH)), color=(255,255,255))
      draw  = ImageDraw.Draw(img)
      pngW  = img.size[0]
      pngH  = img.size[1]
      alpha = pngW/pdfW
      beta  = pngH/pdfH

      BOXLIST = []
      a       = list(map(f5, whatList))
      c       = list(map(f4, a))

      for x in c:
         box = tr_pdftotext_png(x, alpha, beta)
         BOXLIST.append(box)
         draw.rectangle(box, fill=(0,0,0)) #, outline="#80FF00") # grün
   
      if withSave:
         pngfn = PDFfilename + '-' + str(page) + '-'+format +'-' +what + '.png'
         img.save(pathOutput+pngfn)

      return([img, BOXLIST])
 
   ############################################################################# 

   def nextRight(self, m, boxL):
      x1,y1,x2,y2 = m 
      y           = 0.5*(y1 + y2)
      d           = 0.75*abs(y2-y1)
      F1          = list( filter( lambda x: abs( y-x[1])<= d, boxL))
      F1.sort( key=lambda x: x[0] - x1)
      F2          = list( filter( lambda x: x[0] - x1 >0 , F1)) 
      erg         = [] 
      if len(F2)>0:
         erg = F2[0]
     
      return([erg, F1, F2])
  
   ###########################################################################

   def spaceBetweenWords(self, img, imgCheck, boxL, plotBoxes=False, fill=True, uB=20, plotAlsoTooBig= False, xmm=[], bb=10):

      L    = []
      MI   = np.array(imgCheck)[:,:,0]
      SBWL = []
      dd   = ImageDraw.Draw(img)

      for ii in range(len(boxL)):      
         m             = boxL[ii]
         mnext, F1, F2 = self.nextRight(m, boxL)

         if plotBoxes:
            dd.rectangle(m,     width=1, outline="black")   
            dd.rectangle(mnext, width=1, outline="black")         

         if len(mnext)>0:
            mspt     = x1,y1,x2,y2 = m[2], max(m[1], mnext[1]), mnext[0], min( m[3], mnext[3]) 
            checkbox = MI[y1+1:y2-1, x1+1:x2-1]    
            nr, nc   = checkbox.shape

            if np.all(checkbox==255) and nr>0 and nc>0:
               SBWL.append([x1,y1,x2,y2])
            else:
                L.append(mspt)    
         
      borders  = []
      for ii in range(len(xmm)-1):
         a,b = xmm[ii]
         borders.append( [ b-bb,b+bb])      

      DD = []
      for ii in range(len(SBWL)):
         x1,y1,x2,y2 = m = SBWL[ii]
         for jj in range(len(borders)):
            a,b = borders[jj]   
            if x1 <= a <= b <= x2:
               DD.append(ii)
      DD       = list(set(DD))
      SBWL_new = SBWL.copy()

      for jj in range(len(DD)):
         SBWL_new.remove(SBWL[DD[jj]])      

      for ii in range(len(SBWL_new)):
         x1,y1,x2,y2 = m = SBWL_new[ii]
         if fill:
            dd.rectangle([x1, y1, x2, y2], fill="#000000")
         else:
            dd.rectangle([x1, y1, x2, y2], width=3, outline="green")
            
      if plotAlsoTooBig:  
         for ii in range(len(L)):   
            x1,y1,x2,y2 = m = L[ii] 
            dd.rectangle([x1, y1, x2, y2], width=3, outline="red")            
                     
      return([img, L, SBWL, SBWL_new, borders]) 

   #############################################################################   

   def calcSplit(self, img, noc):  # noc=0 gets results from getColMinMax and overrides noc

      C   = np.array(img)
      C1  = np.array( 255*(np.array( C>180, dtype='int')), dtype='uint8')
      
      s   = self.getColumnsCoordinates(C)[0]

      if len(s)!= noc and noc>0:
         msg         = "; calculated columns differ from noc for page"
         self.errors          = self.errors + msg
         _, _, _, start, ende = self.getColumnsCoordinates(C)
         n,m                  = C.shape

         if noc == 1:
            xmm = [ [0, m]]
         if noc == 2:
            r = int((ende-start)/2)
            xmm = [ [0, start + r], [start + r, m]]         
         if noc == 3:
            r = int((ende-start)/3)
            xmm = [ [0, start + r], [start + r, start + 2*r], [start + 2*r, m]]
      else:
         xmm = []
         for col in range(1, noc+1):
            col_x_min, col_x_max = self.getColMinMaxCC(noc, col, C1)
            xmm.append([ col_x_min, col_x_max])

      return(xmm)

   #############################################################################  

   def calcStartAndEnde(self, B):

      B      = np.array(B)
      b      = B.sum(axis=0)/(B.shape[0]*self.white)
      #b      = b.tolist()[0]
      zz     = 0
      while b[zz] == 1 and zz<len(b)-1:
         zz = zz+1

      start = zz

      xx    = len(b)-1
      while b[xx] == 1 and xx>0:
         xx = xx-1
      ende = xx  

      return([start, ende, b])

   ########################################################################

   def getColumnsWindow(self, B, start, ende, fak = 0.97):

      xmms           = []
      mm             = -1
      b              = B.sum(axis=0)/(B.shape[0]*self.white)
      boxL           = []

      if start < ende:     
         mm   = max(b[start:ende])         
         if mm > self.bound:
            found = False
            zz = start
            box = [start]
            while zz < ende:
               if b[zz] >= fak*mm and not(found):
                  found=True
                  box.append(zz)
                  boxL.append(box)
                  box = []
               if found and b[zz] < mm*fak:
                  found=False
                  box = [zz]                               
               zz=zz+1
            if len(box)==1:
               box.append(zz)
               boxL.append(box)
            boxL[-1][1]=ende      

      return([boxL, start, ende, mm, b])
 
   ########################################################################

   def getBorders(self, t,C, start, ende):
      n,m = C.shape
      if len(t)==0:
         t3 = [start, ende]
         t2 = []
      else:
         t1  = list(map( lambda x: np.array(x), t))
         t2  = list(np.median(t1, axis=0))
         t3  = [start] + t2 + [ende]

      return([t3,t2])

   ########################################################################

   def unique(self, L):
      return(  [list(x) for x in set(tuple(x) for x in L)])

   ########################################################################

   def getAllIntersectionIntervals(self, iv, IL):
      a,b = iv
      erg = list(filter(lambda x: (x[0] <= a <= x[1]) or (x[0] <= b <= x[2]), IL))
      return(erg)

   ########################################################################

   def unitedIntervals(self, L):
     
      xL    = []
      N     = []
      zz    = 1
      a,b   = L[0]
      I     = [a]

      while zz < len(L):
         at,bt = L[zz]
         if 0 <= at-b <= self.ub:
            b = bt
         else:
            I.append(b)
            N.append(I)
            a,b = L[zz]
            if zz==len(L)-1:
               N.append([at,bt])
            I = [a]

         zz=zz+1

      N.append([a, b])
      N = self.unique(N)
      N.sort(key=lambda x: x[0])
      
      return(N)
     
   ########################################################################
   
   def getColumnsCoordinates(self, C):
      n,m           = C.shape
      erg           = []
      BL            = []
      Ct            = C[int(n/self.part):int((self.part-1)*n/self.part), :]
      start,ende, b = self.calcStartAndEnde(Ct)

      for ii in range(int(n/self.part), int((self.part-1)*n/self.part)-self.windowSize, self.stepSize):
         B    = np.array(C[ ii: ii + self.windowSize, :])
         BL.append(B)
         xmms, _, _, mm, b = self.getColumnsWindow(B, start=start, ende=ende)         
         if len(xmms)>0:
            xmms              = self.unitedIntervals(xmms)
         erg.append(xmms)

      t1 = list(filter( lambda x: len(x)==1, erg))
      t2 = list(filter( lambda x: len(x)==2, erg)) 
      t3 = list(filter( lambda x: len(x)==3, erg))
      
      def f1(t):
         return( list(map(lambda x: int(x), np.matrix(t).mean(axis=0).tolist()[0])))
      
      lb, ub = self.colRatio
      if len(t3)>0:
         l1, l2, l3 = list(map(lambda x: x[0], t3)), list(map(lambda x: x[1], t3)), list(map(lambda x: x[2], t3))
         c1, c2, c3 = f1(l1), f1(l2), f1(l3)
         a1,b1   = c1
         a2,b2   = c2
         a3,b3   = c3

         if (lb <= (b1-a1)/(b2-a2) <= ub) and (lb <= (b1-a1)/(b3-a3) <= ub) and (lb <= (b3-a3)/(b2-a2) <= ub) and ( ( (b1-a1) + (b2-a2) + (b3-a3))/(b3-a1) >= self.totalRatio) :
            return([[c1,c2,c3], erg, BL, start, ende])

      if len(t2)>0:
         l1, l2 = list(map(lambda x: x[0], t2)), list(map(lambda x: x[1], t2)) 
         c1, c2 = f1(l1), f1(l2)
         a1 ,b1 = c1
         a2, b2 = c2
         
         if lb <= (b1-a1)/(b2-a2) <= ub and ( (b1-a1) + (b2-a2))/(b2-a1) >= self.totalRatio:        
            return([[c1,c2], erg, BL, start, ende])

      return([ [[start, ende]], erg, BL, start, ende])

   ########################################################################

   def getColumns(self, C):

      error              = False
      dLt, erg, BL, _, _ = self.getColumnsCoordinates(C)
      dL                 = []      
      n,m                = C.shape

      if len(dLt)==2:
         a1,b1 = dLt[0]
         a2,b2 = dLt[1]
         dL    = [ min(m, int((a2+b1)*0.5))]

      if len(dLt) == 3:
         a1,b1 = dLt[0]
         a2,b2 = dLt[1]
         a3,b3 = dLt[2]
         dL    = [ int((a2+b1)*0.5), min(m, int((b2+a3)*0.5)) ]

      return(dL)  
      
   #############################################################################      

   def makeCopiesOfColumns(self, C, col, noc):

      if (noc not in (2,3)) or (col not in list(range(1, noc+1))):
         return(C)
     
      D                  = np.ones(C.shape)*255  
      dL                 = self.getColumns(C)
      CC                 = C

      if len(dL)+1 != noc:
         self.errors = "number of columns detected by getColumns() is not the same as given noc"
      else:
         if noc==2:
            Cl, Cr        = D.copy(), D.copy()
            
            if col==1:
               Cl[:, :dL[0]-1] = C[:, :dL[0]-1]
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
               Cl[:, :dL[0]-1]   = C[:, :dL[0]-1]
               Cl_m            = np.roll(Cl, dL[0]-30)
               Cl_r            = np.roll(Cl, dL[1]-20)
               CC              = Cl*Cl_m*Cl_r/(255**2)
            if col==2:
               Cm[:, dL[0]:dL[1]-1] = C[:, dL[0]:dL[1]-1]
               Cm_l               = np.roll(Cm, -(dL[0]-30))
               Cm_r               = np.roll(Cm, dL[0]-30)
               CC                 = Cm*Cm_l*Cm_r/(255**2)
            if col==3:
               Cr[:, dL[1]:]   = C[:, dL[1]:]
               Cr_m            = np.roll(Cr, -(dL[0]-40))
               Cr_l            = np.roll(Cr, -(dL[1]-30))
               CC              = Cr*Cr_m*Cr_l/(255**2)   
 
      return(CC)

   #############################################################################

   def getColMinMaxCC(self, noc, col, C):
   
      col_x_min, col_x_max = 0, C.shape[1]
      dL                   = self.getColumns(C) 

      if len(dL)+1 != noc:
         self.errors = "number of columns detected by getColumns() is not the same as given noc"
      else:

         if col ==1 and noc > 1:
            col_x_min,col_x_max   = 0, dL[0]
         if col ==2 and noc ==2:
            col_x_min, col_x_max = dL[0], C.shape[1]
         if col ==2 and noc ==3:
            col_x_min, col_x_max = dL[0], dL[1]             
         if col ==3:
            col_x_min, col_x_max = dL[1], C.shape[1]
     
      return([col_x_min, col_x_max])

   #############################################################################

   def getsha256(self, path):
      ss = "sha256sum " + path
      tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash').decode('utf-8')   
      return(tt.split(' ')[0])
   
   #############################################################################    

   def cutMatrix(self, imgSBW, xmm, SBWL, fn, fileType, bbm=True):
      C    = np.array(imgSBW)[:,:,0]
      D    = np.array( 255*np.ones(C.shape), dtype='uint8')

      for ii in range(len(xmm)):
         a,b = xmm[ii]
         c   = b-a
         Ct  = D.copy()
         n,m = C.shape
         l   = int(0.5*(m-c))
         Ct[:, l: l+(b-a)] = C[:, a:b]
         img = Image.fromarray(Ct)
         img.name = fn + '-' + str(ii) + fileType
         if bbm:
            img.name = fn + '-bbm-' + str(ii) + fileType
         SBWL.append(img) 

      return(SBWL)

   #############################################################################












class PDF:

   #############################################################################  
   ###  pdfFontSize                                                         ####  
   ###  extractSinglePages                                                  ####  
   ###                                                                      ####  
   #############################################################################  

   def __init__(self, pathPDFFilename,PDFFilename,pathOutput):
      self.pathPDFFilename = pathPDFFilename
      self.PDFFilename     = PDFFilename
      self.pathOutput      = pathOutput
      self.errors          = ""
      self.warnings        = ""
   
   #############################################################################       
        
   def pdfFontSize(self, page):   
      def f4(x):
         return(list(map(float, x)))
      def f5(x):
         return([x["ymin"], x["ymax"]])

      inputfname      = self.PDFFilename + ".pdf"
      outputfname     = self.PDFFilename + "-pdfToText-p" + str(page) + ".xml"
      ss              = "pdftotext -q -bbox-layout -f " + str(page) + " -l " + str(page) + " -htmlmeta " + self.pathPDFFilename + inputfname + " " + self.pathOutput+outputfname; 
      subprocess.check_output(ss, shell=True,executable='/bin/bash')

      soup_pdfToText  = BeautifulSoup(open(self.pathOutput + outputfname), "html.parser")
      whatList        = soup_pdfToText.find_all('word')
   
      a     = list(map(f5, whatList))
      c     = np.median(np.diff(list(map(f4, a))))
      ss    = "rm " + self.pathOutput+outputfname;
      subprocess.check_output(ss, shell=True,executable='/bin/bash')
 
      return(c)
   
   #############################################################################      
      
   def extractSinglePages(self, pageList, outputName=''):
      document = self.pathPDFFilename + self.PDFFilename
      dL = []
      R                 = tqdm(pageList)
      R.set_description('writing single pages as single pdf-files...')
      for page in R:
         dn = document + '-page-' + str(page) + '.pdf'
         ss = 'pdftocairo -q -pdf -f ' + str(page) + ' -l ' + str(page) + ' ' +document + '.pdf ' + dn
         subprocess.check_output(ss, shell=True,executable='/bin/bash')
         dL.append(dn)
   
      ss = 'pdftk '
  
      for docName in dL:
         ss = ss + docName + ' '
      if outputName =='':   
         ss = ss + 'output ' + document+ '_singlePages.pdf'           
      else:
         ss = ss+ 'output ' + self.pathPDFFilename + outputName +'.pdf'
      
      subprocess.check_output(ss, shell=True,executable='/bin/bash')

      for docName in dL:
         ss = 'rm ' + docName
         subprocess.check_output(ss, shell=True,executable='/bin/bash')
         
   #############################################################################      

   def getFormatFromPDFPage(self, page):
      ss = "pdfinfo -f " + str(page) + " -l " + str(page) + " " + self.pathPDFFilename + self.PDFFilename + ".pdf" + " | grep -i '" + str(page) + " size'"  
      tt = subprocess.check_output(ss, shell=True,executable='/bin/bash')
      tt = tt.decode('utf-8')
      tt = tt.replace(' ', '')
      aa = tt.split('x')
      x = float(aa[0].split(':')[1])
      y = float(aa[1].split('pts')[0]) 
   
      format = 'portrait'
      if x > y:
         format = 'landscape'
      
      return([x,y, format])
   
   #############################################################################     
   
   def getNumberOfPagesFromPDFFile(self): 
      ss  = "pdfinfo " + self.pathPDFFilename + self.PDFFilename + ".pdf" + " | grep -i Pages"  
      tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash')
      tt  = tt.decode('utf-8')
      tt  = tt.replace(' ', '').replace('\n','')
      aa  = tt.split(':')
      nOP = int(aa[1]) 
   
      return(nOP)
      
   #############################################################################     













class columns:
   def __init__(self, diff=10, lowB=8, up=100, down=100, white=255):
      self.name     = 'column'
      self.diff     = diff
      self.lowB     = lowB
      self.up       = up
      self.down     = down
      self.white    = white
      self.errors   = ""
      self.warnings = "" 
   
   ################################################################################################

   def getLeftBorder(self, C):
      n,m = C.shape
      D   = C[self.up:n-self.down, :]
      s,t = D.shape

      zz = 0
      fo = False
      r  = D.sum(axis=0)/self.white
   
      while zz < len(r) and not(fo):
         if r[zz]< s:
            fo = True
         else:
            zz = zz+1      

      return(zz)

   ################################################################################################

   def getRightBorder(self, C):
      n,m = C.shape
      D   = C[self.up:n-self.down, :]
      s,t = D.shape

      zz = t-1
      fo = False
      r  = D.sum(axis=0)/255
   
      while zz >= 0 and not(fo):
         if r[zz]< s:
            fo = True
         else:
            zz = zz-1      

      return(zz)

   ################################################################################################

   def getLengthOfSameNeighbourEntries(self, l):

      zz = 1
      ML = []
      for ii in range(len(l)-1):
         if l[ii+1] == l[ii]==self.white:
            zz = zz+1
         else:
            if zz>1:
               ML.append(zz)
            zz = 1
         if l[ii+1] == l[ii]==self.white and ii==len(l)-2:
            ML.append(zz)

      return(ML)
   
   ################################################################################################

   def cutBorders(self, C):

      n,m         = C.shape
      left, right = self.getLeftBorder(C), self.getRightBorder(C)
      D           = C[self.up: n-self.down, left:right]

      return([D, left, right])

   ################################################################################################

   def makeIntervalls(self, M):

      l1 = list(map( lambda x: x[0], M))  
      l1.sort()
 
      iv = []
      IL = []
      zz = 1
      while zz < len(l1):
         if l1[zz]-l1[zz-1] == 1:
            iv.append(l1[zz])
            iv.append(l1[zz-1])
         else:
            iv = list(set(iv))
            iv.sort()
            if len(iv)>0:
               IL.append(iv)
            iv = []   
         zz = zz+1

      if l1[zz-1] - l1[zz-2] == 1:
         iv = list(set(iv))
         iv.sort()
         if len(iv)>0: 
            IL.append(iv)

      IL = list(filter( lambda x: len(x)>= self.lowB, IL))

      return(IL)

   ################################################################################################

   def getMinima(self, ERG):

      ml  =  list(map(lambda x: x[1], ERG))
      a   = list(set(ml))
      a.sort()
      a.reverse()

      if len(a)==1:
         return([a[0], a[0]])
      else:
         return([a[0], a[1]])   

   ################################################################################################

   def coltrane2(self, C):

      Ct             = np.array( 255*np.array( C>200, dtype='int'), dtype='uint8');
      D, left, right = self.cutBorders(Ct)

      ERG = []
      for ii in range(D.shape[1]):
         l   = D[:, ii]
         erg = self.getLengthOfSameNeighbourEntries(l)
         if len(erg)>0:
            ERG.append([ii, max(erg)])
         else:
            ERG.append([ii, -1])

      m,n              = D.shape
      ncols            = 1  
      if len(ERG)>0:
         t1               = n*0.5
         t2,t3            = n*1/3, n*2/3
         m1,m2            = self.getMinima(ERG)
         M1               = list(filter(lambda x: x[1] == m1, ERG))
         IV1              = self.makeIntervalls(M1)
         mp1              = list(map(lambda x: [x[0], x[-1]], IV1))
         s1               = list(map(lambda x: 0.5*(x[1]+x[0]), mp1))
  
         if any(list(map(lambda x: abs(x-t1)< self.diff, s1))):
            ncols = 2

         if len(s1) >1:   
            if any(list(map(lambda x: abs(x-t2)< self.diff, s1))) and any(list(map(lambda x: abs(x-t3)< self.diff, s1))):
               ncols = 3  
         else:
            M2  = list(filter(lambda x: x[1] == m2, ERG))
            IV2 = self.makeIntervalls(M2)
            mp2 = list(map(lambda x: [x[0], x[-1]], IV2))
            s2  = list(map(lambda x: 0.5*(x[1]+x[0]), mp2))
            if (any(list(map(lambda x: abs(x-t2)< self.diff, s1))) or any(list(map(lambda x: abs(x-t3)< self.diff, s1)))) and (any(list(map(lambda x: abs(x-t2)< self.diff, s2))) or any(list(map(lambda x: abs(x-t3)< self.diff, s2))) ):
               ncols = 3

      return([ncols, ERG, left, right, D])

################################################################################################









class stripe:
   def __init__(self, typeOfFile, C, stepSize, windowSize, direction, SWO_2D):
      self.stepSize   = stepSize
      self.windowSize = windowSize
      self.direction  = direction
      self.SWO_2D     = SWO_2D
      self.C          = C
      self.typeOfFile = typeOfFile       

   ########################################################################### 
            
   def displayBoxes(self, img, boxL):

      draw = ImageDraw.Draw(img)

      for ii in range(len(boxL)):
         x1,y1,x2,y2 = boxL[ii]
         draw.rectangle([x1,y1,x2,y2], width=1, outline='red')   
     
      return(img)

   ########################################################################### 

   def makeWL(self, CL, K, xmm, page, hashValue):
 
      WL_B = []
      IMGL = []
      ERGL = [] 

      for ii in range(len(CL)):
         C, col,noc   = CL[ii]
         WL, ergLabel = [],[]
         SS           = self.makeStripes(C)
         Kt           = K.copy()
         n,m          = C.shape
         a,b          = xmm[col]
         c            = b-a
         l            = int(0.5*(m-c))
         d            = l-a 
         if noc>1:
            for jj in range(len(K)):
               hv,x1,y1,x2,y2 = K[jj]
               if  not( a -self.tol <= x1 <= x2 <= b + self.tol):
                  Kt.remove(K[jj])

            for jj in range(len(Kt)):
               hv,x1,y1,x2,y2 = Kt[jj]                  
               Kt[jj]         = hv, x1+d, y1, x2+d, y2
 
         WL, ergLabel = self.labelSS(Kt, SS, hashValue, noc, col, page) 
         WL_B.extend(WL)  
         boxL           = list(map(lambda x: [x[1], x[2], x[3], x[4]], Kt))
         img            = self.displayBoxes(Image.fromarray(C.copy()), boxL)
         IMGL.append(img)

         Cerg         = self.drawErg(C.copy(), WL, ergLabel, a+d,b+d, False)
         img          = Image.fromarray(Cerg)
         ERGL.append(img.copy())         

      return([WL_B, IMGL, ERGL])

   ##############################################################################

   def genMat(self, fname, noc, method, jpggen, onlyWhiteBlack=False, wBB = 180):
 
      #print("generating matrices for each column for method " + INFO.method + "...")

      MAT       = matrixGenerator("downsampling")
      C         = np.matrix( MAT.generateMatrixFromImage(fname + self.typeOfFile), dtype='uint8')
      xmm       = jpggen.calcSplit(C, noc)
   
      if method=='bBHV':
         fname = fname + '-bbm'
         C     = np.matrix( MAT.generateMatrixFromImage(fname+ self.typeOfFile), dtype='uint8')     

      CL  = []
      if noc==1:
         if onlyWhiteBlack:
            C = np.array( 255*np.array( C>= wBB, dtype='int'), dtype='uint8')
         CL.append([C, 0,1] )
      else:   
         for col in range(0, noc):
            C = np.matrix( MAT.generateMatrixFromImage(fname+ '-' + str(col) + self.typeOfFile), dtype='uint8')
            if onlyWhiteBlack:
               C = np.array( 255*np.array( C>= wBB, dtype='int'), dtype='uint8')
            CL.append([C, col, noc])

      return([CL, xmm])

################################################################################################

   def makeStripes(self, C):         
       
      n,m        = C.shape
      ii, SS     = 0, []
      #print("windowsize:" + str(self.windowSize))

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
         
      return(SS)
    
   ###########################################################################
 
   def ergForBox(self,ss, box, erg):     
      
      ws              = self.windowSize
      box_x1, box_x2  = box[1], box[3]   # box_x1 <= box_x2
      box_y1, box_y2  = box[2], box[4]   # box_y1 <= box_y2
           
      if self.direction == 'H':
         ya, ye   = ss[1], ss[2]    # ya <= ye
                    
         if (box_y1 <= ya) and (box_y2 >= ye):  
            erg = 1
         if box_y1> ya and (box_y1 -ya)/ws <= self.dd:
            erg = 1 
         if box_y2 < ye and (ye-box_y2)/ws <= self.dd:
            erg = 1    
      
      if self.direction == 'V':
         xa, xe  = ss[1], ss[2]
           
         if (box_x1 <= xa) and (box_x2 >= xe):  
            erg = 1
         if box_x1>  xa and (box_x1 -xa)/ws <= self.dd:
            erg = 1 
         if box_x2<  xe and (xe-box_x2)/ws <= self.dd:
            erg = 1 
                
      return(erg)  
    
   ###########################################################################      
         
   def labelSS(self, K, SS, hashValue, noc, col, page):
      WL     = []
      for ss in SS:
         erg    = 0
         for box in K:
            erg = self.ergForBox(ss, box, erg)            

         WL.append([ss[0], [ss[1], ss[2]], hashValue, page, col, noc, erg])

      ergLabel = list(map(lambda x: x[6], WL))
   
      return([WL, ergLabel])
         
   ###########################################################################
      
   def generateLabelBoxes(self, kindOfBox, hashValue, con):
      
      WL, ergLabel = [], []

      if kindOfBox == 'TA':
         if self.typeOfFile == '.jpg':
            SQL    = "select hashValueJPGFile, LUBOX_x, LUBOX_y, RLBOX_x, RLBOX_y from boxCoordinatesOfTables where hashValueJPGFile = '" + hashValue + "'"   
         if self.typeOfFile == '.png':
            SQL    = "select hashValueJPGFile, LUBOX_x, LUBOX_y, RLBOX_x, RLBOX_y from boxCoordinatesOfTables where hashValuePNGFile = '" + hashValue + "'"   
      else:   
         if self.typeOfFile == '.jpg':
            SQL    = "select hashValueJPGFile, LUHL_x, LUHL_y, RLHL_x, RLHL_y     from boxCoordinatesOfTables where hashValueJPGFile = '" + hashValue + "' and LUHL_x is not NULL"   
         if self.typeOfFile == '.png':
            SQL    = "select hashValueJPGFile, LUHL_x, LUHL_y, RLHL_x, RLHL_y     from boxCoordinatesOfTables where hashValuePNGFile = '" + hashValue + "' and LUHL_x is not NULL"   

      rs     = con.execute(SQL)
      K      = list(rs)   

      return(K)

   ###########################################################################
   
   def removeZeroImages(CL, upTo=79):
      B   = list( filter( lambda x: sum(np.array(x[0])[0:upTo]) == 0, CL))
      bl  = list( map   ( lambda x: x[2], B))
      CLt = list( filter( lambda x: x[2] not in bl, CL))
   
      return(CLt)

   ###########################################################################

   def prepareData(self, WL, des):
      CLt = list(map(lambda x: x[0], WL))
      la  = list(map(lambda x: x[6], WL))
      CL  = []
      MAT = matrixGenerator("downsampling")

      for cl in CLt:
         A = MAT.downSampling(cl, self.downSamplingRate )
         B = MAT.adaptMatrix(A, self.adaptMatrixCoef[0], self.adaptMatrixCoef[1])
         CL.append(B)
   
      ERG = MISC.makeIt(CL, self.SWO_2D, des)   
      
      return([ERG, la])  

   ###########################################################################    

   def drawErg(self, C, WL, erg, xmin,xmax, hp=False):
      
      for ii in range(len(WL)):
         wl = WL[ii]
         #if erg[ii]==1:
         a,b = wl[1][0], wl[1][1]

         if self.direction == 'H':
            if not(hp):
               C[a, :] = 0
               C[b-1:b+1, :] = 0
            else:
               C[a, xmin:xmax] = 0
               C[b-1:b+1, xmin:xmax] = 0

         if self.direction == 'V':
            if xmin <= a <= xmax:
               C[:, a] = 0
            if xmin <= b <= xmax:   
               C[:, b-1:b+1] = 0
               
      return(C)    

###########################################################################










class imageGeneratorPNG:

   def __init__(self, pathToPDF, pdfFilename, outputFolder, output_file, pageStart, pageEnd, scanedDocument = False, windowSize=450, stepSize=50, bound=0.99, part=8, ub=5, size=(595, 842) ):
      self.pathToPDF       = pathToPDF
      self.pdfFilename     = pdfFilename
      self.pageStart       = pageStart
      self.pageEnd         = pageEnd
      self.outputFolder    = outputFolder
      self.outputFile      = output_file
      self.scanedDocument  = scanedDocument
      self.L               = []
      self.size            = size
      self.IMOP            = imageOperations(windowSize, stepSize, bound, part, ub)
      self.PDF             = PDF(pathToPDF, pdfFilename, outputFolder)
      self.errors          = ""
      self.warnings        = ""

   #############################################################################

   def saveImg(self, IMGL, SBWL):

      for ii in range(len(IMGL)):
         img = IMGL[ii]
         fn  = img.name
         img.save(fn)

      for ii in range(len(SBWL)):
         img = SBWL[ii]
         fn  = img.name
         img.save(fn)

   ###############################################################################

   def findPageNOC(self, NOCL, page):
      noc = 0
      for ii in range(len(NOCL)):
         l = NOCL[ii]
         if l[1] == page:
            noc = l[2]
      
      return(noc)

   #############################################################################

   def generateSinglePNG(self, page, fn, noc):

      img4,BOXLIST               = self.IMOP.pdfToBlackBlocksAndLines(self.pathToPDF, self.pdfFilename, self.outputFolder, page, 'word', 'portrait', withSave=True, useXML = False)
      if noc == -1:
         C               = np.array(img4)[:,:,0] 
         dT, _, _, _, _  = self.IMOP.getColumnsCoordinates(C)
         noc             = len(dT)
         #print(noc)

      self.IMOP.errors           = "" 
      xmm                        = self.IMOP.calcSplit(np.array(img4)[:,:,0], noc) 
      
      BOXLIST                    = list(map( lambda x: list(np.round(np.array(x),0)), BOXLIST))
      BOXLIST                    = list(map( lambda x: list(map(lambda y: int(y) , x)), BOXLIST))

      img4.save(fn + '.png') 
      img4.name                  = fn+'.png'
      IMGL                       = [img4]
      
      imgSBW                     = Image.new(mode="RGB",size=self.size, color=(255,255,255))
      imgSBW, _, _, _, _         = self.IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img4, boxL=BOXLIST, plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
      imgSBW.name                = fn + '-bbm.png'    
      SBWL                       = [imgSBW] 

      if noc>1:
         SBWL = self.IMOP.cutMatrix(imgSBW, xmm, SBWL, fn, '.png')
         IMGL = self.IMOP.cutMatrix(img4, xmm, IMGL, fn, '.png', False)     

      return([SBWL, IMGL])

   #############################################################################
 
   def generate(self, getNOCfromDB=True, onlyScanned=False):

      MAT             = matrixGenerator('downsampling')
      MAT.description = "genPNG" 
      N               = []
      self.Q          = []
      SQL             = "SELECT namePDFDocument, page, numberOfColumns FROM TAO where namePDFDocument='" + self.pathSQL + self.pdfFilename  + "' and format='portrait' and what='word' and original='YES' and page in ("
      NOCL            = []

      if self.pageStart ==1 and self.pageEnd == 0:
         self.pageEnd = self.PDF.getNumberOfPagesFromPDFFile()
      
      if len(self.L)==0:   
         R = tqdm(range(self.pageStart, self.pageEnd+1))
         for ii in range(self.pageStart, self.pageEnd+1):
            SQL = SQL + str(ii) + ','
      else :
         for ii in range(len(self.L)):
            SQL = SQL + str(self.L[ii]) + ','
         R = tqdm(self.L) 
      SQL = SQL[0:-1] + ') order by page'
      
      if getNOCfromDB:
         rs   = self.con.execute(SQL)  
         NOCL = list(rs) 

      R.set_description('generating PNGs in ' + self.outputFolder + ' ...')
      
      for page in R:                                                   
         x,y,format = self.PDF.getFormatFromPDFPage(page)   
            
         if format == 'portrait':
            fn         = self.outputFolder + self.outputFile +'-' + str(page) +'-'+format + '-' + 'word'
            noc = -1
            if getNOCfromDB:
               noc     = self.findPageNOC(NOCL, page)
           
            SBWL, IMGL = self.generateSinglePNG(page, fn, noc)
            self.saveImg(IMGL, SBWL)

            hashPNG = self.IMOP.getsha256(fn+ '.png')
         
            if (not hashPNG in list(map(lambda x: x[6], N))):
               atime = datetime.now()
               dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
               erg = [self.pathToPDF + self.pdfFilename, fn+ '.png', None, format, 'word', page, hashPNG, None, dstr]
               N.append(erg)    

      self.A = pd.DataFrame( N, columns=['namePDFDocument', 'filenamePNG', 'filenameJPG','format',  'what', 'page', 'hashValuePNGFile', 'hashValueJPGFile', 'timestamp'])

   ################################################################################################

   def wordsInLine(self, L, WL, worddist):
    
      CL = []
      cl = [0] 
      for ii in range(0,len(L)-1):
         x1,y2,x2,y2 = box1 = L[ii][0]
         s1,t1,s2,t2 = box2 = L[ii+1][0]
         if s1-x2 < worddist:
            cl.append(ii+1) 
            if ii+1==len(L)-1:
               CL.append(cl)
         else:
            CL.append(cl) 
            cl = [ii+1]
            if ii+1== len(L)-1:
               CL.append(cl)
 
      return(CL)

   #############################################################################

   def concatString(self, ss):
      erg = ""
      for ii in range(len(ss)):
         erg = erg + ss[ii] + " "
      erg = erg[0:-1]

      return(erg)     

   ################################################################################################

   def groupingInLine(self, page, worddist=6):

      A, BOXLIST        = self.IMOP.pdfToBlackBlocksAndLines(self.pathToPDF, self.pdfFilename, self.outputFolder, page, 'word', 'portrait', withSave=True, useXML = False)
      outputfname       = self.pdfFilename + "-pdfToText-p" + str(page) + ".xml"
      soup_pdfToText    = BeautifulSoup(open(self.outputFolder + outputfname), "html.parser")
      whatList          = soup_pdfToText.find_all('word')
      WL = []
      for wl in whatList:
         box1 = [ wl['xmin'], wl['ymin'], wl['xmax'], wl['ymax']]
         box2 = list(map( lambda x: int(np.round(float(x),0)), box1))
         WL.append( [box2, wl.text])

      lines = list(set(list(map(lambda x: x[0][3], WL))))
      lines.sort()
      LINES = []
      for ll in lines:
         L = list(filter(lambda x: x[0][3] ==ll , WL))
         L.sort(key=lambda x: x[0])
         CL =[[0]] 
         if len(L) >1:
            CL = self.wordsInLine(L, WL, worddist)
         T  = []
         for ii in range(len(CL)):
            cl, cs = list(map(lambda x: L[x][0], CL[ii])), list(map(lambda x: L[x][1], CL[ii]))
            if len(cl) >1:
               T.append( [ [ cl[0][0], cl[0][1], cl[-1][2], cl[-1][3]], self.concatString(cs) ])
            else:
               T.append( [  cl[0], cs[0] ])
         LINES.append(T)

      LINES = list(filter(lambda x: len(x)>0, LINES))

      return([LINES, WL])    

################################################################################################








class imageGeneratorJPG:

   def __init__(self, pathToPDF, pdfFilename, outputFolder, output_file, pageStart, pageEnd, scanedDocument = False, dpi=200, generateJPGWith='tesseract', windowSize=450, stepSize=50, bound=0.99, part=8, ub=5, size=(595, 842) ):
      self.pathToPDF       = pathToPDF
      self.pdfFilename     = pdfFilename
      self.pageStart       = pageStart
      self.pageEnd         = pageEnd
      self.dpi             = dpi
      self.outputFolder    = outputFolder
      self.outputFile      = output_file
      self.scanedDocument  = scanedDocument
      self.generateJPGWith = generateJPGWith
      self.L               = []
      self.size            = size
      self.IMOP            = imageOperations(windowSize, stepSize, bound, part, ub)
      self.PDF             = PDF(pathToPDF, pdfFilename, outputFolder)
      self.errors          = ""
      self.warnings        = ""

      if not(scanedDocument):
         self.imageGeneratorPNG = imageGeneratorPNG( pathToPDF, pdfFilename, outputFolder, output_file, pageStart, pageEnd, scanedDocument = False, size=(595, 842) )

   #############################################################################      
   
   def makeMatrixFromImage(self, imgOrg, tolW= 0.3, tolH= 0.2, debug=False, W=595, H=842):

      itd         = pytesseract.image_to_data(imgOrg, output_type=Output.DICT)
      medh        = np.median( itd['height'])
      boxL        = []
      boxLOrg     = []
      W_org,H_org = imgOrg.width, imgOrg.height
      alpha       = H/H_org

      img     = Image.new(mode="RGB",size=(W, H), color=(255,255,255))
      draw    = ImageDraw.Draw(img)
      n_boxes = len(itd['level'])      
      for i in range(n_boxes):
         if itd['level'][i] == 5 and len(itd['text'][i].replace(' ', ''))>0:
            (x, y, w, h) = (itd['left'][i], itd['top'][i], max(10, itd['width'][i]), max(itd['height'][i],10))    
            co = [alpha*x, alpha*y, alpha*(x+w), alpha*(y+h)]
            co = list(map( lambda x: round(x,0), co))
            co = list(map(int, co))
            boxLOrg.append(co) 
            if w <= W*tolW and 3*medh >= h >= tolH*medh:
               boxL.append(co) 
               if debug:
                  draw.rectangle([alpha*x, alpha*y, alpha*(x+w), alpha*(y+h)], width=3, outline="red")
               else: 
                  draw.rectangle([alpha*x, alpha*y, alpha*(x+w), alpha*(y+h)], fill="#000000")

      M    = np.array(img)[:,:,0]
      imgM = Image.fromarray(M)
         
      return([imgM, M, img, itd, imgOrg, boxL, boxLOrg])
   
   #############################################################################     

   def reduceBoxes(self, N, h=10, w=3):
      l         = list(set(list( map(lambda x: x[1], N)))) 
      l.sort()
      k  = [l[0]]
      zz = 1
      while zz < len(l):
         if abs(l[zz]-k[-1]) > h:
            k.append(l[zz])
         zz = zz+1
   
      M = []
      for ii in range(len(k)):
         m = list( filter(lambda x:  0 <= x[1]-k[ii]<= h, N))
         m.sort(key=lambda x: x[0])
         M.append(m)

      Mt = []
      for ii in range(0,len(M)):
         m  = M[ii]
         mt = []
         jj = 1
         boxalt = m[jj-1]
         while jj < len(m):
            boxneu = m[jj] 
            if (boxalt[2] >=  boxneu[0] or abs(boxalt[2] -boxneu[0]) <= w) and (jj< len(m)-1): 
               boxalt = [ min(boxalt[0], boxneu[0]), min( boxalt[1], boxneu[1]), max(boxneu[2], boxalt[2]), max(boxalt[3], boxneu[3])]   
            else:   
               mt.append(boxalt.copy())      
               #print("ii=" + str(ii) + " boxalt=" + str(boxalt))
               boxalt = boxneu
            jj = jj+1 
         Mt.append(mt)
  
      Mtn = [x for x in Mt if x != []]
      Mtt = []
      for ii in range(len(Mtn)):
         mt    = Mtn[ii]
         ymin  = min( list(map(lambda x: x[1], mt)))
         ymax  = max( list(map(lambda x: x[3], mt)))   
         mtt   = [] 
         for jj in range(len(mt)):
            m = mt[jj]
            mtt.append([m[0], ymin, m[2], ymax]) 
         Mtt.append(mtt)

      Mtt_n     = []
      for ii in range(len(Mtt)):
         mtt  = Mtt[ii]
         mttc = mtt.copy()
         for jj in range(len(mtt)):
            x1,y1,x2,y2 = box1 = mtt[jj]
            for kk in range(jj+1, len(mtt)):
               v1,w1,v2,w2 = box2 = mtt[kk]      
               if x2 >= v2 >= v1 >= x1 and y2 >= w2 >= w1 >= y1:
                  mttc.remove(box2)
         Mtt_n.append(mttc)
   
      return(Mtt_n)
 
   #############################################################################   

   def makeBlackBox(self, imgOrg, radius_edge=1, radius_blur=3, sigma=0.15, hb=20, wb=20, hb2=10, wb2=3):

      M2 = np.array(imgOrg)
      C  = np.array( 255*np.array(M2 >180, dtype='int'), dtype='uint8')
      ny = Wimage.from_array(np.matrix(C))

      ny.edge(radius_edge)
      ny.blur(radius=radius_blur, sigma=sigma)
      A1     = np.array(ny)[:,:,0]
      B1     = np.array( 255*np.array(A1 < 1, dtype='int'), dtype='uint8')   

      contours, hierachy = cv.findContours(B1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      img1    = Image.new(mode="RGB",size=(595,842), color=(255,255,255))
      draw1   = ImageDraw.Draw(img1)
      img2    = img1.copy()
      draw2   = ImageDraw.Draw(img2)
 
      R       = range(len(contours))
      boxL    = []
      boxLOrg = []
      for ii in R:
         c                                   = contours[ii]
         Next, Previous, First_Child, Parent = hierachy[0,ii,:] 
         x, y, w, h                          = cv.boundingRect(c)
         boxLOrg.append( [x, y, x+w, y+h])
         if h <= hb and w<=wb:
            boxL.append( [x, y, x+w, y+h])

      Mtt = []
      if len(boxL)>0:
         Mtt = self.reduceBoxes(boxL, h=hb2, w=wb2)
         for ii in range(0,len(Mtt)):
            m = Mtt[ii]
            for jj in range(1,len(m)): 
               x1,y1,x2,y2 = m[jj-1]
               v1,w1,v2,w2 = m[jj]    
               draw2.rectangle([x2,y1,v1,w2], fill="#000000")
               draw2.rectangle([x2,y1,v1,w2], width=1, outline="red") 
      
         for ii in range(0,len(Mtt)):
            m = Mtt[ii]
            for jj in range(0,len(m)): 
               x1,y1,x2,y2 = m[jj]
               draw1.rectangle([x1,y1,x2,y2], fill="#000000")
               #draw1.rectangle([x2,y1,v1,w2], width=1, outline="red") 

      MttN  = [y for x in Mtt for y in x]

      return([img1,img2, MttN, boxLOrg])   

   #############################################################################     

   def wordsInLine(self, L, WL, worddist):
    
      CL = []
      cl = [0] 
      for ii in range(0,len(L)-1):
         x1,y2,x2,y2 = box1 = L[ii][0]
         s1,t1,s2,t2 = box2 = L[ii+1][0]
         if s1-x2 < worddist:
            cl.append(ii+1) 
            if ii+1==len(L)-1:
               CL.append(cl)
         else:
            CL.append(cl) 
            cl = [ii+1]
            if ii+1== len(L)-1:
               CL.append(cl)
 
      return(CL)

   #############################################################################

   def concatString(self, ss):
      erg = ""
      for ii in range(len(ss)):
         erg = erg + ss[ii] + " "
      erg = erg[0:-1]

      return(erg)     

   ################################################################################################

   def makeG(self, lines, WL):
 
      d     = np.diff(lines)
      G     = []
      g     = [ WL[0]]
      for ii in range(len(d)):
         if d[ii] <= 3:
            g.append(WL[ii+1])    
            if ii == len(d)-1:
               G.append(g)  
         else:
            g.sort(key=lambda x: x[0][0])
            G.append(g)
            g = [ WL[ii+1]]

      for ii in range(len(G)):
         g = G[ii]
         r = list(map(lambda x: x[0][1], g))[0]
         for jj in range(len(g)):
            g[jj][0][1] = r

      return(G)

   ################################################################################################

   def makeH(self, lines, G, worddist):
      H= []
      for ii in range(len(G)):
         g        = G[ii]
         lines    = [ g[0]]
         if len(g)==1:
            H.append(lines)
         else: 
            box1, txt1 = g[0]
            for jj in range(1,len(g)):
               box2, txt2 = g[jj]
               x1,y1,x2,y2 = box1
               s1,t1,s2,t2 = box2
               if s1-x2 < worddist:
                  lines.append(g[jj])
                  box1, txtx1 = g[jj]
               else:
                  H.append(lines)
                  box1, txtx1 = g[jj]
                  lines = [ g[jj]]
               if jj==len(g)-1:
                  H.append(lines)
      return(H)

   ################################################################################################

   def makeH_GLUE(self, H):

      H_GLUE = []
      for ii in range(len(H)):
         w, boxList = list(map(lambda x: x[1], H[ii])), list(map(lambda x: x[0], H[ii]))
         ss         = self.concatString(w)
         box        = [ boxList[0][0], boxList[0][1], boxList[-1][2], boxList[-1][3]] 
         H_GLUE.append( [box, ss]) 
   
      return(H_GLUE)

   ################################################################################################

   def groupingInLine(self, page, worddist=6):
   
      global alpha
   
      def f(x):
         global alpha
        
         a = np.round( alpha*np.array(x))
         b = list( map(int, a))
         return(b)
      
      pages   = self.convertPDFToJPG(page, page)      
      img     = Image.open(pages[0])
      
      H, W    = img.height, img.width
      alpha   = 842/H
      itd     = pytesseract.image_to_data(image=img, output_type=Output.DICT, lang='deu')
      
      ss = "rm " + pages[0]
      tt = subprocess.check_output(ss, shell=True,executable='/bin/bash')
      
      left, top, width, height = list(map( f, [ itd['left'], itd['top'], itd['width'],itd['height'] ]))
      text                     = itd['text']
      
      WL = []
      for ii in range(len(text)):
         if len(text[ii].replace(' ', '')) >0:
            x1,y1,x2,y2 = left[ii], top[ii], left[ii] + width[ii], top[ii] + height[ii]
            WL.append([ [x1,y1,x2,y2], text[ii] ])

      WL.sort(key=lambda x: x[0][1])
      lines = list(map(lambda x: x[0][1], WL))
      #lines.sort()

      G      = self.makeG(lines, WL)
      H      = self.makeH(lines, G, worddist)
      H_GLUE = self.makeH_GLUE(H)
      
      LINES  = []
      lines  = list(set(list(map(lambda x: x[0][1] , WL))))
      lines.sort()
      LINES  = list(map(lambda x: list(filter(lambda y: y[0][1]==x , H_GLUE)) , lines))

      return([LINES, WL])      
          
   ############################################################################# 
   
   def convertPDFToJPG(self, firstPage, lastPage):
   
      pages = convert_from_path( pdf_path     = self.pathToPDF+ self.pdfFilename+ ".pdf", 
                                         dpi  = self.dpi, 
                               output_folder  = self.outputFolder+ 'tmp', 
                                         fmt  = 'jpeg', 
                                 output_file  = self.outputFile, 
                                  first_page  = firstPage, 
                                   last_page  = lastPage, 
                                  paths_only  = True)       
      
      return(pages)
          
   #############################################################################        
     
   def deleteFromMatrix(self, M, d, w=255):

      J1     = w*np.array( M==d, dtype='int')
      J2     = M*np.array( M!=d, dtype='int')
      J3     = np.array( J1+J2, dtype='uint8')

      return(J3)   
   
   ###########################################################################

   def makeCheckMatrix(self, imgOrg, size, p=0.01, ws=25, bb=250):

      imgT   = imgOrg.resize(size)                
      MT     = np.array(imgT)
      J3     = MT.copy()
      n,m    = J3.shape
      zz     = 1
   
      while zz*ws < n:
         W = J3[(zz-1)*ws:zz*ws, :]
         h,t    = np.histogram(W, bins = list(range(0, 257)))
         r      = np.round(h/sum(h),3)
         s      = t[np.where(r >= p)]
         for ii in range(len(s)):  
            W = self.deleteFromMatrix(W, s[ii])
         J3[(zz-1)*ws:zz*ws, :] = W
         zz = zz+1

      B1     = np.array( 255*np.array(J3 > bb, dtype='int'), dtype='uint8')   

      return(B1)

   #############################################################################
 
   def makeQualityCheck(self, img1, img2):
   
      M1   = np.array(img1)[:,:,0]
      M2   = np.array(img2)[:,:,0] 
      n,m  = min(M1.shape[0], M2.shape[0]), min(M1.shape[1], M2.shape[1])         
      M1   = M1[0:n, 0:m] 
      M2   = M2[0:n, 0:m] 

      l1  = list(np.concatenate(M1.tolist())) 
      l2  = list(np.concatenate(M2.tolist()))
      M12I= np.matrix( 255*np.matrix( M1+M2 > 0, dtype='int'), dtype='uint8') 
      l12i= list(np.concatenate(M12I.tolist()))
      if l1.count(0) == 0 or l2.count(0) == 0:
         q =  [np.inf, np.inf]
      else:
         p1, p2 = l12i.count(0)/l1.count(0), l12i.count(0)/l2.count(0)
         q      = [  p1, p2 ]

      return(q) 

   #############################################################################

   def deleteBox(self, draw, B, boxL, dL, b=0.3):
      for ii in range(0,1*len(boxL)):
          x1,y1,x2,y2 = m = boxL[ii]
          F           = B[y1:y2, x1:x2]  
          l           = list(np.concatenate(F.tolist()))
          if len(l)>0:
             if l.count(0)/len(l) < b:
                m = [x1-1, y1-1, x2+1, y2+1]
                draw.rectangle(m, fill="#FFFFFF")   
                dL.remove(boxL[ii])

      return([draw, dL])

   #############################################################################

   def addBox(self, draw, B, boxL, aL, b = 0.3, c = 50):
      for ii in range(0,len(boxL)):
         x1,y1,x2,y2 = boxL[ii]  
         F           = B[y1:y2, x1:x2] 
         l           = list(np.concatenate(F.tolist()))
         if l.count(0)/len(l) < b and abs(x1-x2)*abs(y1-y2)>= c:
            draw.rectangle(boxL[ii], fill="#000000")   
            aL.extend([boxL[ii]])

      return([draw, aL])

   #############################################################################
  
   def displayBoxes(self, boxL, minL=15, color='red', fill=True):

      img  = Image.new(mode="RGB",size=(595,842), color=(255,255,255))
      draw = ImageDraw.Draw(img)
      cbL  = []

      for ii in range(len(boxL)):
         x1,y1,x2,y2 = boxL[ii]
         if abs((x1-x2)*(y1-y2)) >= minL:
            if not(fill):
               draw.rectangle([x1,y1,x2,y2], width=1, outline=color)   
            else:
               draw.rectangle([x1,y1,x2,y2], fill=(0,0,0))
 
            cbL.append([x1,y1,x2,y2])

      return([img, cbL])

   ###############################################################################

   def saveImg(self, IMGL, SBWL):

      for ii in range(len(IMGL)):
         img = IMGL[ii]
         fn  = img.name
         img.save(fn)

      for ii in range(len(SBWL)):
         img = SBWL[ii]
         fn  = img.name
         img.save(fn)

   ###############################################################################

   def generateSBW(self, q, img1, img3,  NNL, boxL2, noc, xmm, fn):

      pp, p1,p2, p3,p4 = q
      imgSBW           = Image.new(mode="RGB",size=self.size, color=(255,255,255))
      
      if not(self.scanedDocument):
         if p3 < 0.7 and p1 >= p2:
            imgSBW, L, SBWLL, SBWLL_new, b = self.IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img1, boxL=NNL,   plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
         else:
            imgSBW, L, SBWLL, SBWLL_new, b = self.IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img3, boxL=boxL2, plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
      else:
         if p1>= p2:
            imgSBW, L, SBWLL, SBWLL_new, b = self.IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img1, boxL=NNL,   plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
         else:
            imgSBW, L, SBWLL, SBWLL_new, b = self.IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img3, boxL=boxL2, plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)

      imgSBW.name = fn + '-bbm.jpg'    
      SBWL        = [imgSBW]

      if noc>1:
         SBWL = self.IMOP.cutMatrix(imgSBW, xmm, SBWL, fn, '.jpg')

      return(SBWL)

   #############################################################################    

   def imageToSave(self, q):
 
      pp, p1,p2, p3,p4 = q
      imgStr           = 'img3'

      if not(self.scanedDocument):
         if p3 < 0.7 and p1 >= p2:         
            imgStr    = 'img1'
      else:
         if p1>= p2:
            imgStr    = 'img1'
 
      return(imgStr)
 
   ########################################################################

   def generateJPGOnePage(self, imgOrg, format, fn, page, noc):

      imgT                       = imgOrg.resize(self.size)
      img1, img1t, NN,  N_org    = self.makeBlackBox(imgT, radius_edge=1, radius_blur=3, sigma=0.15, hb=10, wb=20, hb2=4, wb2=8)                   
      _, _, img2, _, _, boxL2, _ = self.makeMatrixFromImage(imgOrg, tolW=1, tolH=0) 
      img3                       = img2.copy()
      draw3                      = ImageDraw.Draw(img3)
      B1                         = self.makeCheckMatrix(imgOrg, size=self.size)
      draw3, boxL2               = self.deleteBox(draw3, B1, boxL2, boxL2.copy())     
      M3                         = np.array(img3)[:,:,0] 
      draw3, boxL2               = self.addBox(draw3, M3, NN, boxL2)
      q                          = [page] + self.makeQualityCheck(img1, img2)  
     
      if len(q)==3:
         q.extend([0,0])
      self.Q.append(q)
      
      xmm       = []
      imgToSave = self.imageToSave(q)
      if imgToSave == 'img3':
         img3.name = fn + '.jpg'
         IMGL      = [img3]
         C         = np.array(img3)[:,:,0]
      else:
         img1.name = fn + '.jpg'
         IMGL      = [img1]
         C         = np.array(img1)[:,:,0]
   
      if noc == -1: 
         dT, _, _, _, _  = self.IMOP.getColumnsCoordinates(C)
         noc             = len(dT)
        #print(noc)

      if noc>1:         
         D   = np.array( 255*np.ones(C.shape), dtype='uint8')
         xmm = self.IMOP.calcSplit(C, noc)

         for ii in range(len(xmm)):
            a,b = xmm[ii]
            c   = b-a
            Ct  = D.copy()
            n,m = C.shape
            l   = int(0.5*(m-c))
            Ct[:, l: l+(b-a)] = C[:, a:b]
            img = Image.fromarray(Ct)
            img.name = fn + '-' + str(ii) + '.jpg'
            IMGL.append(img) 
      
      SBWL = self.generateSBW(q, img1, img3, NN, boxL2, noc, xmm, fn) 
    
      return([IMGL, img1, img2, img3, SBWL, NN, boxL2, q])

   #############################################################################

   def findPageNOC(self, NOCL, page):
      noc = 0
      for ii in range(len(NOCL)):
         l = NOCL[ii]
         if l[1] == page:
            noc = l[2]
      
      return(noc)

   #############################################################################

   def generate(self, getNOCfromDB=True, onlyScanned=False):

      MAT                  = matrixGenerator('JPGGen')
      MAT.description      = "MatGen"
      N                    = []
      self.Q               = []
      SQL                  = "SELECT namePDFDocument, page, numberOfColumns FROM TAO where namePDFDocument='" + self.pathToPDF + self.pdfFilename  + "' and format='portrait' and what='word' and original='YES' and page in ("
      NOCL                 = []
      ss                   = "rm -f " + self.outputFolder+ "tmp/*.jpg"

      if self.pageStart ==1 and self.pageEnd == 0:
         self.pageEnd = self.PDF.getNumberOfPagesFromPDFFile()
      
      if len(self.L)==0:   
         R = tqdm(range(self.pageStart, self.pageEnd+1))
         for ii in range(self.pageStart, self.pageEnd+1):
            SQL = SQL + str(ii) + ','
      else :
         for ii in range(len(self.L)):
            SQL = SQL + str(self.L[ii]) + ','
         R = tqdm(self.L) 

      SQL = SQL[0:-1] + ') order by page'
      
      if getNOCfromDB:
         rs   = self.con.execute(SQL)  
         NOCL = list(rs)

      R.set_description('generating JPG in ' + self.outputFolder + ' ...')
      
      for page in R:
         tt         = subprocess.check_output(ss, shell=True,executable='/bin/bash')  
         pages      = self.convertPDFToJPG(page, page)                                                       
         x,y,format = self.PDF.getFormatFromPDFPage(page)   
            
         if format == 'portrait': 
            imgOrg  = Image.open(pages[0]).convert('L')
            fn      = self.outputFolder + self.outputFile +'-' + str(page) +'-'+format + '-' + 'word'
            noc     = -1
            if getNOCfromDB:
               noc = self.findPageNOC(NOCL, page)
           
            IMGL, img1, img2, img3, SBWL, NN, boxL2, q   = self.generateJPGOnePage(imgOrg, format, fn, page, noc)
            self.saveImg(IMGL, SBWL)

            if not(self.scanedDocument):
               _, IMGL    = self.imageGeneratorPNG.generateSinglePNG(page, fn, noc=1)
               self.saveImg(IMGL, [])
               hashPNG    = self.IMOP.getsha256(fn+ '.png')
               hashJPG    = self.IMOP.getsha256(fn+ '.jpg')
         
               if (not hashJPG in list(map(lambda x: x[7], N))) and (not hashPNG in list(map(lambda x: x[6], N))):
                  atime = datetime.now()
                  dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
                  erg = [self.pathToPDF + self.pdfFilename, hashJPG, fn+ '.jpg', format, 'word', page, hashPNG, hashJPG, dstr]
                  N.append(erg)    

            else:
               hashJPG = self.IMOP.getsha256(fn + '.jpg')
               if not hashJPG in list(map(lambda x: x[7], N)):
                  atime = datetime.now()
                  dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
                  erg = [self.pathToPDF + self.pdfFilename, None, fn + '.jpg', format, 'word', page, 'scanedDocument', hashJPG, dstr]
                  N.append(erg)                

      self.A = pd.DataFrame( N, columns=['namePDFDocument', 'filenamePNG', 'filenameJPG','format',  'what', 'page', 'hashValuePNGFile', 'hashValueJPGFile', 'timestamp'])

   #############################################################################      



