
import sys, subprocess
from os import system
import os
sys.path.append('/home/markus/anaconda3/python/modules')
import csv
from datetime import datetime 
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import timeit
from PIL import Image, ImageDraw
import pandas as pd

import mysql.connector                     # pip install mysql-connector-python
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
from bs4 import BeautifulSoup
import pickle
import os.path

from functools import partial
from joblib import Parallel, delayed
import multiprocessing as mp

import pywt

import scatteringTransformationModule_2D_v9 as ST
import misc_v9 as MISC

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import copy

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

###########################################################################

def printMat(A):
   print('\n'.join([''.join(['|{:5}'.format(item) for item in row]) for row in A]))

###########################################################################

def evaluateCalibration(pred, labels):
   r    = list(range(1, len(pred)+1))
   erg  = list(1*(pred == labels)) 
   M    = np.array([ list(r), list(pred), list(erg), list(labels)], dtype='int')
   printMat(tp(M))
   print(erg.count(0)/len(erg))

#############################################################################

def saveIt(ERG, fname):
   
   a     = datetime.now()
   dstr  = a.strftime("%d.%m.%Y-%H:%M:%S") 
   pickle_out = open(fname + '-'+dstr, 'wb')
   pickle.dump(ERG, pickle_out)
   pickle_out.close()
   return(dstr)

#############################################################################         
        
def loadIt(fname):
   pickle_in   = open(fname,"rb")
   CL          = pickle.load(pickle_in)   
   return(CL)        
          
#############################################################################         

def f(i):
   def fi(x):
      return(x[i])
   return(fi)
 
f0 = f(0)
f1 = f(1)
f2 = f(2)
f3 = f(3)
f4 = f(4)
f5 = f(5)

#############################################################################  

def pdfToBlackBlocksAndLines(pathPDFFilename, PDFfilename, pathOutput, page, what, format, withSave=True, useXML = True):   

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

   
   inputfname      = PDFfilename + ".pdf"
   outputfname     = PDFfilename + "-pdfToText-p" + str(page) + ".xml"
   
   if not(useXML):   
      ss              = "pdftotext -q -bbox-layout -f " + str(page) + " -l " + str(page) + " -htmlmeta " + pathPDFFilename + inputfname + " " + pathOutput+outputfname; 
      subprocess.check_output(ss, shell=True,executable='/bin/bash')

   #print(ss)
   soup_pdfToText  = BeautifulSoup(open(pathOutput + outputfname), "html.parser")
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
      pngfn = PDFfilename + '-' + str(page) + '-'+format +'-' +what + '.png'
      img.save(pathOutput+pngfn)

   return(img)
 
############################################################################# 

def getFormatFromPDFPage(pathPDFFilename, PDFFilename, page):

   ss = "pdfinfo -f " + str(page) + " -l " + str(page) + " " + pathPDFFilename+ PDFFilename + ".pdf" + " | grep -i '" + str(page) + " size'"  
   tt = subprocess.check_output(ss, shell=True,executable='/bin/bash')
   tt = tt.decode('utf-8')
   tt = tt.replace(' ', '')
   aa = tt.split('x')
   x = float(aa[0].split(':')[1])
   y = float(aa[1].split('pts')[0]) 
   
   return([x,y])

#############################################################################  
  
def getNumberOfPagesFromPDFFile(pathPDFFilename, PDFFilename):

   ss  = "pdfinfo " + pathPDFFilename+ PDFFilename + ".pdf" + " | grep -i Pages"  
   tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash')
   tt  = tt.decode('utf-8')
   tt  = tt.replace(' ', '').replace('\n','')
   aa  = tt.split(':')
   nOP = int(aa[1]) 
   
   return(nOP)
   
#############################################################################

def getsha256(path, file):
   ss = "sha256sum " + path+ file  
   tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash').decode('utf-8')   
   return(tt.split(' ')[0])
   
#############################################################################

def makeEven(x):
   return( x + x%2)    
 
#############################################################################

def makeIt(CL,SWO,description ):
   
   tt = tqdm(CL)
   tt.set_description_str('calculation SWCs for ' + description  +'...')
   
   t1 = timeit.time.time() 
   foo_ = partial(ST.deepScattering, SWO=SWO)
   output = Parallel(mp.cpu_count())(delayed(foo_)(i) for i in tt)
   t2 = timeit.time.time(); print(t2-t1)
   return(output)            

#############################################################################

def findPos(l,L):
   try:
      pos = L.index(l)
   except:
      pos = -1   

   return(pos) 

#############################################################################

def prepareDataForRF(whichLabelToPredict, SWC, SQL, con):

   rs  = con.execute(SQL)
   HL  = []
   for row in rs:
      rr = list(row)
      for ii in range(len(rr)):
         if rr[ii]==None:
            rr[ii] = ''
      HL.append(rr)
   
   AL = []
   al = [] 
   pl = []

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
   
   return([AL, al, pl])

############################################################################# 

def makeValidationSet(whichLabelToPredict, SWC):
   AL_challenge = []
   al_challenge = []
   pl_challenge = []   
   O            = getattr(SWC.challenge.labels, whichLabelToPredict)
   
   for ii in range(len(SWC.challenge.data.HL)):
      h    = SWC.challenge.data.HL[ii] 
      lpos = findPos(h, O.HL)
      if lpos>=0:
         AL_challenge.append(SWC.challenge.data.AL[ii])
         al_challenge.append(O.LL[lpos])
         pl_challenge.append(h)    

   return([AL_challenge, al_challenge, pl_challenge])

############################################################################# 
#############################################################################  









class matrixGenerator:

   #############################################################################  
   ###  padding                | adaptMatrix       | generateOnlyMatrices   ####  
   ###  maxC                   | generateSWC       | compressMatrix         ####  
   ###  generateMatrixFromPNG  | generateOnlySWC   | getLabels              ####
   ###  downSampling           | generateMatrices  |                        ####  
   #############################################################################


   def __init__(self, compressMethod):
      self.compressMethod = compressMethod
   
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
          
#############################################################################  










class PNGGenerator:

   #############################################################################  
   ###  generatePNGS          | inserDataIntoDB                             ####  
   ###  extractSinglePages    |                                             ####  
   ###  insertRawDataIntoDB   |                                             ####  
   #############################################################################  

   def __init__(self, pathPDFFilename, PDFFilename, DB='TAO', table='TAO', deleteXML = True):  # PDFFilename without ending .pdf, i.e. if your file is named "document.pdf" then PDFFilename='document'
      self.PDFFilename     = PDFFilename
      self.pathPDFFilename = pathPDFFilename
      self.DB              = DB
      self.table           = table
      self.deleteXML       = deleteXML

   #############################################################################

   def generatePNGS(self, pathOutput, what, startPage=1, endPage=0 ) :
      self.pathOutput = pathOutput
      self.startPage  = startPage
      self.endPage    = endPage
      self.what       = what
      
      if endPage ==0:
         endPage = getNumberOfPagesFromPDFFile(self.pathPDFFilename, self.PDFFilename)

      self.pathOutputTotal = pathOutput + what + '/'
      N = []
      R = tqdm(range(startPage, endPage+1))
      R.set_description('generating PNGS in ' + self.pathOutputTotal + ' ...')
         
      for page in R:
      
         [x,y] = getFormatFromPDFPage(self.pathPDFFilename, self.PDFFilename, page)   
         format = 'portrait'
         if x > y:
            format = 'landscape'
            
         pngfn = self.pathOutputTotal + self.PDFFilename +'-' + str(page) +'-'+format + '-' + self.what + '.png'
         pdfToBlackBlocksAndLines( self.pathPDFFilename, self.PDFFilename, self.pathOutputTotal, page, what, format, True, False)
         
         hash = getsha256(self.pathOutputTotal, self.PDFFilename +'-' + str(page) +'-'+format + '-' + self.what + '.png')
         
         if not hash in list(map(f5, N)):
            atime = datetime.now()
            dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
            erg = [self.pathPDFFilename + self.PDFFilename, pngfn, format, self.what, page, hash, dstr]
            N.append(erg)
         else:
            ss = 'rm ' +  pngfn
            subprocess.check_output(ss, shell=True,executable='/bin/bash')
              
      self.A = pd.DataFrame( N, columns=['namePDFDocument', 'filenamePNG', 'format',  'what', 'page', 'hashValuePNGFile', 'timestamp'])
      
      if self.deleteXML:
         oldpath = os.getcwd()
         os.chdir(self.pathOutputTotal)
         ss = 'rm *.xml'
         subprocess.check_output(ss, shell=True,executable='/bin/bash')
         os.chdir(oldpath)
      
   #############################################################################   
      
   def insertRawDataIntoDB(self, user, passwd):
      ss     = 'mysql+pymysql://' + user + ':' + passwd + '@localhost/' + self.DB
      engine = create_engine(ss)
      self.A.to_sql(self.table+'_tmp', engine, if_exists='append', index=False)
      
      con    = engine.connect()
      COL   = list(con.execute('select * from ' + self.table).keys())
      COLS  = ''
      for col in COL:
         COLS = COLS + col + ','
      COLS = COLS[0:-1]    
      
       
      SQL    = 'DELETE FROM ' + self.table+'_tmp where hashValuePNGFile in (select hashValuePNGFile from ' + self.table + ')' 
      rs     = con.execute(SQL)
      SQL    = "INSERT INTO " + self.table + "(" + COLS + ") select " + COLS + " from " + self.table + "_tmp"
      rs     = con.execute(SQL)
      SQL    = "DELETE FROM " + self.table+"_tmp"
      rs     = con.execute(SQL)
      con.close()
      
      #SQL    = "INSERT INTO " + self.table + "(namePDFDocument, filenamePNG, format, what, page, hashValuePNGFile, timestamp) select namePDFDocument, filenamePNG, format, what, min(page)," 
      #SQL    = SQL + " hashValuePNGFile, timestamp from " + self.table + "_tmp group by namePDFDocument, filenamePNG, format, what, hashValuePNGFile, timestamp"
      
   #############################################################################   
      
   def insertDataIntoDB(self, user, passwd, DB, table, A):
      engine = create_engine('mysql+pymysql://' + user + ':' + passwd +'@localhost/' + DB)
      con    = engine.connect()

      A.to_sql(table + '_tmp', engine, if_exists='replace', index=False)
      COL   = list(con.execute('select * from ' + table).keys())
      COLS  = ''
      for col in COL:
         COLS = COLS + col + ','
      COLS = COLS[0:-1]           

      SQL    = "UPDATE " + table + " INNER JOIN " + table + "_tmp on " + table + ".hashValuePNGFile = " + table + "_tmp.hashValuePNGFile SET " 
      SQL    = SQL + table + ".hasTable              = "+ table + "_tmp.hasTable,"  
      SQL    = SQL + table + ".numberOfColumns       = "+ table + "_tmp.numberOfColumns,"
      SQL    = SQL + table + ".col1                  = "+ table + "_tmp.col1,"
      SQL    = SQL + table + ".col2                  = "+ table + "_tmp.col2,"
      SQL    = SQL + table + ".col3                  = "+ table + "_tmp.col3,"
      SQL    = SQL + table + ".pageConsistsOnlyTable = "+ table + "_tmp.pageConsistsOnlyTable,"
      SQL    = SQL + table + ".source                = "+ table + "_tmp.source,"
      SQL    = SQL + table + ".timestamp             = "+ table + "_tmp.timestamp,"
      SQL    = SQL + table + ".T1                    = "+ table + "_tmp.T1,"
      SQL    = SQL + table + ".T2                    = "+ table + "_tmp.T2,"
      SQL    = SQL + table + ".T3                    = "+ table + "_tmp.T3,"
      SQL    = SQL + table + ".T4                    = "+ table + "_tmp.T4,"
      SQL    = SQL + table + ".numberOfTables        = "+ table + "_tmp.numberOfTables;"
      rs     = con.execute(SQL)
     
      SQL    = "DELETE FROM " + table + "_tmp where hashValuePNGFile in (select hashValuePNGFile from " + table + ");"
      rs     = con.execute(SQL)
     
      SQL    = "INSERT INTO " + table + "(" + COLS + ") select " + COLS + " from " + table + "_tmp;"
      rs     = con.execute(SQL)
     
      SQL    = "DELETE FROM " + table + "_tmp;"
      rs     = con.execute(SQL)
   
      con.close()
    
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










class RF:

   #############################################################################  
   ###  fitRF                  | standardize                                ####  
   ###  shuffleDataRF          | normalize                                  ####  
   ###  evaluateRF             |                                            ####
   ###  getOverviewAnnotations |                                            ####  
   #############################################################################
   
   def __init__(self, name, AL, al, pl, useStandardized=False, info='', verbose=0, numberOfTrees=500 ):
      self.name            = name
      self.AL_org          = AL
      self.al              = al
      self.pl              = pl
      self.info            = info
      self.useStandardized = useStandardized
      self.verbose         = verbose
      self.numberOfTrees   = numberOfTrees
      self.standardize()
      
      if useStandardized:
         self.AL = self.AL_st.copy()
         if verbose>0:
            print("using standardized data...")
      else:
         self.AL = self.AL_org.copy()
         if verbose >0:
            print("using non-standardized data ...")
         
      self.fitRF()
      self.shuffleDataRF()
      
      
   #############################################################################
   
   def fitRF(self):
      self.rf          = RandomForestClassifier(n_estimators=self.numberOfTrees)
      self.rf.fit(self.AL, self.al)
      
   #############################################################################
      
   def shuffleDataRF(self):
      def f0(x):
         return(x[0])   
      def f1(x):
         return(x[1])
      def f2(x):
         return(x[2])
      
      R  = []
      for ii in range(len(self.AL)):
         R.append([self.AL[ii], self.al[ii], self.pl[ii]])
      np.random.shuffle(R)
  
      self.AL = list(map(f0, R))
      self.al = list(map(f1, R))
      self.pl = list(map(f2, R))
   
   ############################################################################# 
        
   def evaluateRF(self, V, v, asDict=False):
   
      print(self.name)
      
      Vt = V.copy()
      if self.useStandardized:
         Vt = self.normalize(V)
         
      ypred = self.rf.predict(Vt)
      erg   = metrics.classification_report(ypred, v, output_dict= asDict)
      
      self.ypred = ypred
      
      if asDict:
         return(erg)
      else:
         print(erg)   
         print("length AL: " + str(len(self.AL)))
         print("length V: " + str(len(V)))
         return(erg)     
     
   #############################################################################  
      
   def getOverviewOfAnnotations(self):
      al = self.al   
      aa = list(set(al))
      aa.append(max(aa)+1)
      M  = np.matrix(np.histogram(al, aa))
      m1 = list(M[0,1].flatten())
      m2 = list(M[0,0].flatten())
      m1.remove(max(m1))
      M =  mat([m1,m2])
      return(M)    
 
   #############################################################################  
 
   def standardize(self, vvr=0.001):
      AL                 = np.matrix(self.AL_org)
      v                  = tp(tp(AL).var(1))
      z                  = np.array(v> vvr , dtype='int')
      Z                  = np.zeros((AL.shape[1],AL.shape[1])); np.fill_diagonal(Z, z)
      #idx                = np.argwhere(np.all(Z[..., :] == 0, axis=0))
      #Zt                 = np.delete(Z, idx, axis=1)
      #ALt                = AL.dot(Zt)
      ALt                = AL
      M                  = np.tile(tp(tp(ALt).mean(1)), (ALt.shape[0],1) )
      vt                 = tp(tp(ALt).var(1))
      c                  = list(np.array(vt).flatten())
      V                  = 1/sqrt(np.tile(vt, (ALt.shape[0],1) ))
      B                  = np.round( np.array((ALt-M))*np.array(V), 2)
      T                  = list(map(list, list(B)))

      mt                 = M[0, :]
      self.mt            = mt
      self.vt            = vt  
      self.AL_st         = T
      
   ############################################################################# 
   
   def normalize(self, v):
      erg = (v - self.mt)/sqrt(self.vt)
      return(erg)   
  
#############################################################################  
   





   
   
   
   
class compressMethods:
   def __init__(self, name, level, adaptMatrix, wavename='bior2.6'):
      self.name            = name  
      self.MAT             = matrixGenerator('downsampling')   
      self.MAT.level       = level
      self.MAT.aM          = adaptMatrix
      self.wavename        = wavename
      self.xroll           = -5
      self.yroll           = -10
   #############################################################################
      
   def downsampling(self,C):
      MAT            = self.MAT
      MAT.mm, MAT.nn = MAT.aM[0], MAT.aM[1]
      C1 = MAT.downSampling(C, MAT.level)
      C2 = MAT.adaptMatrix(C1, MAT.mm, MAT.nn)
      
      return(C2)
      
   #############################################################################   
    
   def pywt1(self, C):
      coeffs   = pywt.wavedec2(C, self.wavename, level=3)
      Ct       = np.roll(coeffs[0], self.xroll)
      Ct       = np.roll(Ct,self.yroll, axis=0)
      Ct       = self.MAT.adaptMatrix(Ct, 106, 76)
   
      return(Ct)
      
#############################################################################  










class cnnModel:
   def __init__(self, name, model, level, adaptMatrix):
      self.name        = name
      self.model       = model
      self.level       = level
      self.adaptMatrix = adaptMatrix
      self.MAT         = matrixGenerator("gaga")
        
   #############################################################################     
        
   def prepareMatrix(self, C):
      C1 = self.MAT.downSampling(C, self.level, padding=True)
      C2 = self.MAT.adaptMatrix(C1, self.adaptMatrix[0], self.adaptMatrix[1])
      CL  = list(np.array([C2])/255)
      erg = np.array(CL).reshape([len(CL), self.adaptMatrix[0], self.adaptMatrix[1],1])
      return(erg)

   #############################################################################

   def predictClass(self, C):
      Ct  = self.prepareMatrix(C)
      erg = self.model.predict_classes(Ct)
      return(erg[0]) 
 
#############################################################################  













class columns:
   def __init__(self, diff=10, lowB=8, up=100, down=100, white=255):
      self.name  = 'column'
      self.diff  = diff
      self.lowB  = lowB
      self.up    = up
      self.down  = down
      self.white = white
   
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
   def __init__(self, C, stepSize, windowSize, direction, SWO_2D):
      self.stepSize   = stepSize
      self.windowSize = windowSize
      self.direction  = direction
      self.SWO_2D     = SWO_2D
      self.C          = C

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
      C         = np.matrix( MAT.generateMatrixFromImage(fname+'.jpg'), dtype='uint8')
      xmm       = jpggen.calcSplitJPGs(C, noc)
   
      if method=='bBHV':
         fname = fname + '-bbm'
         C     = np.matrix( MAT.generateMatrixFromImage(fname+'.jpg'), dtype='uint8')     

      CL  = []
      if noc==1:
         if onlyWhiteBlack:
            C = np.array( 255*np.array( C>= wBB, dtype='int'), dtype='uint8')
         CL.append([C, 0,1] )
      else:   
         for col in range(0, noc):
            C = np.matrix( MAT.generateMatrixFromImage(fname+ '-' + str(col) + '.jpg'), dtype='uint8')
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
         SQL    = "select hashValueJPGFile, LUBOX_x, LUBOX_y, RLBOX_x, RLBOX_y from boxCoordinatesOfTables where hashValueJPGFile = '" + hashValue + "'"   
      else:   
         SQL    = "select hashValueJPGFile, LUHL_x, LUHL_y, RLHL_x, RLHL_y     from boxCoordinatesOfTables where hashValueJPGFile = '" + hashValue + "' and LUHL_x is not NULL"   

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











class JPGNPNGGenerator:

   def __init__(self, pathToPDF, pdfFilename, outputFolder, output_file, pageStart, pageEnd, scanedDocument = False, dpi=200, generateJPGWith='tesseract', size=(595, 842) ):
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

   #############################################################################     
   
   def getsha256(self, path):
      ss = "sha256sum " + path
      tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash').decode('utf-8')   
      return(tt.split(' ')[0])
   
   #############################################################################     
   
   def getFormatFromPDFPage(self, page):
      ss = "pdfinfo -f " + str(page) + " -l " + str(page) + " " + self.pathToPDF + self.pdfFilename + ".pdf" + " | grep -i '" + str(page) + " size'"  
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
      ss  = "pdfinfo " + self.pathToPDF + self.pdfFilename + ".pdf" + " | grep -i Pages"  
      tt  = subprocess.check_output(ss, shell=True,executable='/bin/bash')
      tt  = tt.decode('utf-8')
      tt  = tt.replace(' ', '').replace('\n','')
      aa  = tt.split(':')
      nOP = int(aa[1]) 
   
      return(nOP)
      
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
   
   def groupingInLine(self, page, diff=4):
   
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
      
      level = 5
      M     = tp([ itd['level'], left, top, width, height, list(range(len(text))) ])
      P     = np.array( list( filter( lambda x: x[0] == level and len(itd['text'][x[5]].replace(' ','')) >0, M)))

      A  = P.copy()
      A  = A[:, 1:]
      A = A[ A[:, 1].argsort()]

      LINES = []
      l     = []
      R     = list(set( A[:, 1]))
      R.sort()
   
      T = R.copy()
      d = np.diff(R)
      r = d.argsort()
      for ii in range(len(r)):
         if d[r[ii]] <diff:
            T.remove(R[r[ii]])

      for ii in range(len(T)):
         y = T[ii]
         N = np.array( list( filter(lambda x: abs(x[1]-y)<= diff, A) ))       
         N = N[ N[:, 0].argsort()]
         LINES.append(N)

      L  = []
      wg = []

      for ii in range(len(LINES)):
         wg   = []
         line = LINES[ii].tolist()
   
         wg.append(line[0])
         for jj in range(len(line)-1):
            (x, y, w, h, tx1) = line[jj]         
            (r, s, t, u, tx2) = line[jj+1]        
            if abs(x + w- r) <= diff:
               wg.append(line[jj+1])
            else:
               L.append(wg)
               wg = [ line[jj+1] ]
         L.append(wg)
   
      BL = []
      for ii in range(len(L)):
         R         = L[ii]
         x1,y1,w,h,tx1 = R[0]   
         x, y, w,h,tx2 = R[len(R)-1] 
         x2,y2     = x+w, y+h
         tx             = ""
         for jj in range(len(L[ii])):
            tx = tx + " " +text[R[jj][4]]
         BL.append([x1,y1,x2,y2,tx])      

      return(BL)      
          
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

   def spaceBetweenWordsJPG(self, img, imgCheck, boxL, plotBoxes=False, fill=True, uB=20, plotAlsoTooBig= False, xmm=[], bb=10):

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
            mspt     = x1,y1,x2,y2 = m[2], max(m[1], mnext[1]), mnext[0], min( m[3], mnext[3]) #    m[2], m[1], mnext[0], mnext[3]
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

   #############################################################################

   def generateSBW(self, q, img1, img3,  NNL, boxL2, noc, xmm, fn):

      pp, p1,p2, p3,p4 = q
      imgSBW           = Image.new(mode="RGB",size=self.size, color=(255,255,255))
      
      if not(self.scanedDocument):
         if p3 < 0.7 and p1 >= p2:
            imgSBW, L, SBWLL, SBWLL_new, b = self.spaceBetweenWordsJPG(img=imgSBW, imgCheck=img1, boxL=NNL,   plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
         else:
            imgSBW, L, SBWLL, SBWLL_new, b = self.spaceBetweenWordsJPG(img=imgSBW, imgCheck=img3, boxL=boxL2, plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
      else:
         if p1>= p2:
            imgSBW, L, SBWLL, SBWLL_new, b = self.spaceBetweenWordsJPG(img=imgSBW, imgCheck=img1, boxL=NNL,   plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)
         else:
            imgSBW, L, SBWLL, SBWLL_new, b = self.spaceBetweenWordsJPG(img=imgSBW, imgCheck=img3, boxL=boxL2, plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)

      imgSBW.name = fn + '-bbm.jpg'    
      SBWL        = [imgSBW]

      if noc>1:
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
            img.name = fn + '-bbm-' + str(ii) + '.jpg'
            #img.show()
            SBWL.append(img) 

      #Image.fromarray(C).show()
      return(SBWL)

   #############################################################################    

   def getColumns(self, C, ws=50, st=3):

      n,m       = C.shape
      zz        = st
      S         = np.zeros( (m))
      while zz*ws + 150 <= n:
         W = C[(zz-1)*ws:zz*ws, :]
         w = np.round(W.sum(axis=0)/(255*ws),2)
         S = S+w
         zz = zz+1

      S  = S/(zz-st)
      S1 = np.round(S,1)

      first = True
      for ii in range(len(S1)):
         if S1[ii] == 1 and first:
            S1[ii] = 0
         else:
            if S1[ii] <1:
               first = False

      first = True
      xx = m-1
      while xx >0:
         if S1[xx] == 1 and first:
            S1[xx] = 0
         else:
            if S1[xx] <1:
               first = False
         xx = xx-1

      lfB = True
      xx = 0
      L  = []
      l  = []
      while xx <= len(S1)-1:
         if S1[xx] == 1 and lfB:
            l= [xx]
            lfB = False
         else:
            if S1[xx] <1 and not(lfB):
               if xx-l[0] >=10:
                  l.append(xx)
                  L.append(l)
               l = []
               lfB=True

         xx = xx+1      
 
      L  = list( map( lambda x: int(0.5*(x[0] + x[1])),L ))

      return(L)  
      
   #############################################################################      

   def makeCopiesOfColumns(self, C, col, noc):

      if (noc not in (2,3)) or (col not in list(range(1, noc+1))):
         return(C)
     
      D  = np.ones(C.shape)*255   
      dL = self.getColumns(C)        

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
 
   ########################################################################

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

   def getColMinMaxCC(self, noc, col, C):
   
      col_x_max = C.shape[1]
      col_x_min = 0     
         
      if col>0:
         dL        = self.getColumns(C)
         
         if noc>1:
            if len(dL)>0:
               CC        = self.makeCopiesOfColumns(C, col, noc)
               if dL[0]> C.shape[1]/2:
                  diff = int( dL[0]- (C.shape[1]/2))
                  CC   = np.roll(CC, -diff)
         else:
            CC = C

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
         CC = C
         print("CC = C")
         
      
      return([CC, col_x_min, col_x_max])

   #############################################################################

   def generateJPGSNPNGSOnePage(self, imgOrg, format, fn, page, noc):

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
     
      img4                       = img1.copy()

      if not(self.scanedDocument):
         img4      = pdfToBlackBlocksAndLines( self.pathToPDF, self.pdfFilename, self.outputFolder, page, 'word', format, False, False)  
         img4.save(fn + '.png')       
         q2        =  self.makeQualityCheck(img3, img4)
         q.extend(q2)
    
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
   
      if noc>1:         
         D   = np.array( 255*np.ones(C.shape), dtype='uint8')
         xmm = self.calcSplitJPGs(C, noc)

         #if len(xmm)==0:
         #   print(xmm)
         #   print("error with page=" + str(page))
         #else:
         #   if xmm[0][1] == 0:
         #      print(xmm)
         #      print("error with page=" + str(page)) 

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
    

      return([IMGL, img1, img2, img3, img4, SBWL, NN, boxL2, q])

   #############################################################################

   def findPageNOC(self, NOCL, page):
      noc = 0
      for ii in range(len(NOCL)):
         l = NOCL[ii]
         if l[1] == page:
            noc = l[2]
      
      return(noc)

   #############################################################################

   def calcSplitJPGs(self, img, noc):

      C   = np.array(img)
      C1  = np.array( 255*(np.array( C>180, dtype='int')), dtype='uint8')

      xmm = []
      for col in range(1, noc+1):
         CC, col_x_min, col_x_max = self.getColMinMaxCC(noc, col, C1)
         xmm.append([ col_x_min, col_x_max])

      return(xmm)

   #############################################################################

   def generateJPGSNPNGS(self, getNOCfromDB=True, onlyScanned=False):

      N          = []
      self.Q     = []
      SQL        = "SELECT namePDFDocument, page, numberOfColumns FROM TAO where namePDFDocument='" + self.pathToPDF + self.pdfFilename  + "' and format='portrait' and what='word' and original='YES' and page in ("
      NOCL       = []
      ss         = "rm -f " + self.outputFolder+ "tmp/*.jpg"

      if self.pageStart ==1 and self.pageEnd == 0:
         self.pageEnd = self.getNumberOfPagesFromPDFFile()
      
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

      else:
         if len(self.L)>0:
            NOCL = np.ones(len(self.L))
         else:
            NOCL = np.ones(self.pageEnd-self.pageStart)

      R.set_description('generating JPGs and PNGs in ' + self.outputFolder + ' ...')
      
      for page in R:
         tt         = subprocess.check_output(ss, shell=True,executable='/bin/bash')  
         pages      = self.convertPDFToJPG(page, page)                                                       
         x,y,format = self.getFormatFromPDFPage(page)   
            
         if format == 'portrait': 
            noc                                                = self.findPageNOC(NOCL, page)
            print("page=" + str(page) + " noc=" + str(noc))
            imgOrg                                             = Image.open(pages[0]).convert('L')
            fn                                                 = self.outputFolder + self.outputFile +'-' + str(page) +'-'+format + '-' + 'word'
            IMGL, img1, img2, img3, img4, SBWL, NN, boxL2, q   = self.generateJPGSNPNGSOnePage(imgOrg, format, fn, page, noc)
            self.saveImg(IMGL, SBWL)

            if not(self.scanedDocument):
               hashJPG = self.getsha256(fn+ '.jpg')
               hashPNG = self.getsha256(fn+ '.png')
         
               if not hashJPG in list(map(lambda x: x[7], N)) and not hashPNG in list(map( lambda x: x[6], N)):
                  atime = datetime.now()
                  dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
                  erg = [self.pathToPDF + self.pdfFilename, fn+ '.png', fn+ '.jpg', format, 'word', page, hashPNG, hashJPG, dstr]
                  N.append(erg)    

            else:
               hashJPG = self.getsha256(jpgfn)
               if not hashJPG in list(map(lambda x: x[7], N)) :
                  atime = datetime.now()
                  dstr  = atime.strftime("%Y-%m-%d %H:%M:%S")
                  erg = [self.pathToPDF + self.pdfFilename, None, fn + '.jpg', format, 'word', page, 'scanedDocument', hashJPG, dstr]
                  N.append(erg)                

      self.A = pd.DataFrame( N, columns=['namePDFDocument', 'filenamePNG', 'filenameJPG','format',  'what', 'page', 'hashValuePNGFile', 'hashValueJPGFile', 'timestamp'])

   #############################################################################      

   def insertDataIntoDBGeneric(self, user, passwd, DB, table, A):
      engine = create_engine('mysql+pymysql://' + user + ':' + passwd +'@localhost/' + DB)
      con    = engine.connect()
      SQL    = "DELETE FROM " + table + "_tmp"
      rs     = con.execute(SQL)
     
      A.to_sql(table + '_tmp', engine, if_exists='replace', index=False)
      COL   = list(con.execute('select * from ' + table+"_tmp").keys())
      COLS  = ''
      for col in COL:
         COLS = COLS + col + ','
      COLS = COLS[0:-1]           
      
      SQL0      = "select A.COLUMN_NAME from ( "
      SQL1      = "select tab.table_schema as database_schema, sta.index_name as pk_name, sta.seq_in_index as column_id, sta.column_name, tab.table_name "
      SQL2      = "from information_schema.tables as tab inner join information_schema.statistics as sta on sta.table_schema = tab.table_schema and sta.table_name = tab.table_name "
      SQL3      = "and sta.index_name = 'primary' where tab.table_schema = 'TAO' and tab.table_type = 'BASE TABLE' "
      SQL4      = ") A where A.TABLE_NAME = '" + table + "'"
      SQL       = SQL0 + SQL1 + SQL2 + SQL3 + SQL4
      
      rs        = con.execute(SQL)
      LLL       = list(rs)
      
      NB        = ""
      NB_fields = list( map(lambda x: x[0], LLL))
      for ii in range(len(NB_fields)):
         name = NB_fields[ii]
         NB = NB + "a."+ name + " = b." + name 
         if ii< len(NB_fields)-1:   
            NB = NB + ' AND '

      print("updating on " + NB)
   
      SQL    = "UPDATE " + table + " a INNER JOIN " + table + "_tmp b on " + NB
      SQL    = SQL + " SET " 
      for ii in range(len(COL)):  
         SQL    = SQL + "a." + COL[ii] + " = b." + COL[ii]  
         if ii < len(COL)-1:
            SQL = SQL + ","
      rs     = con.execute(SQL)
      
      SQL    = "DELETE FROM " + table + "_tmp where hashValuePNGFile in (select hashValuePNGFile from " + table + ") and hashValueJPGFile in (select hashValueJPGFile from " + table + ");"
      rs     = con.execute(SQL)
           

      SQL    = "INSERT INTO " + table + "(" + COLS + ") select " + COLS + " from " + table + "_tmp;"
      rs     = con.execute(SQL)
     
      SQL    = "DELETE FROM " + table + "_tmp;"
      rs     = con.execute(SQL)
      
      con.close()


