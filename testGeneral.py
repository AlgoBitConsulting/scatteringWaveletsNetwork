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
 
from os import listdir
from os.path import isfile, join

###################################################################################################################

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real

###################################################################################################################
pathPDFFilename = '/home/markus/anaconda3/python/pngs/train/'
PDFFilename     = 'train'
pathJPGs        = '/home/markus/anaconda3/python/pngs/train/test/'
pathPNGs        = '/home/markus/anaconda3/python/pngs/train/word/'
pathOutput      = '/home/markus/anaconda3/python/pngs/train/test/'


def getColumns2(C, white=255):
   n,m            = C.shape
   B              = C[int(n/3):2*int(n/3), :]
   b = B.sum(axis=0)/(B.shape[0]*white)

   zz = 0
   while b[zz] == 1:
      zz = zz+1

   start = zz
   xx    = len(b)-1
   while b[xx] == 1:
      xx = xx-1
   ende = xx  

   mm   = max(b[start:ende])
   xmms = []
   if mm >0.95:
      found = False
      while zz< xx:
         if b[zz] == mm and not(found):
            found=True
            xmms.append(zz)
         if found and b[zz]<mm:
            found=False  
         zz=zz+1

   return([xmms, start, ende, mm, b])


PDF             = dOM.PDF(pathPDFFilename, PDFFilename, pathOutput)
IMOP            = dOM.imageOperations()
MAT             = dOM.matrixGenerator('downsampling')
MAT.description = "TEST"
#C1              = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-216-portrait-word.png')

#a               = PDF.pdfFontSize(10)
#b               = PDF.extractSinglePages([10,100])
#format          = PDF.getFormatFromPDFPage(100)
nOP             = PDF.getNumberOfPagesFromPDFFile()
#imgSBW          = Image.new(mode="RGB",size=(595, 842), color=(255,255,255))
#img1            = Image.fromarray(C1)      
#xmm             = IMOP.calcSplitJPGs(C1, 2)

#img, NNL        = IMOP.pdfToBlackBlocksAndLines(pathPDFFilename, PDFFilename, pathOutput, 216, 'word', 'portrait', withSave=True, useXML = False)

#NNL             = list(map( lambda x: list(np.round(np.array(x),0)), NNL))
#NNL             = list(map( lambda x: list(map(lambda y: int(y) , x)), NNL))
#imgSBW, L, SBWLL, SBWLL_new, b = IMOP.spaceBetweenWords(img=imgSBW, imgCheck=img, boxL=NNL,   plotBoxes=False, fill=True, uB=800, plotAlsoTooBig=False, xmm= xmm)

#img.save('/home/markus/anaconda3/python/pngs/train/test/test.png')
#hashPNG  = IMOP.getsha256('/home/markus/anaconda3/python/pngs/train/test/test.png')


trainJPG         = dOM.imageGeneratorJPG(pathPDFFilename, PDFFilename, pathJPGs, 'train', 1, 5, True, 500, 'cv')     
trainJPG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
trainJPG.con     = trainJPG.engine.connect()
trainJPG.L       = [113,115]  
#trainJPG.generateJPG()

trainPNG         = dOM.imageGeneratorPNG(pathPDFFilename, PDFFilename, pathPNGs, 'train', 1, 0, False)     
trainPNG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
trainPNG.con     = trainPNG.engine.connect()
trainPNG.generatePNG()


"""

C1             = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-32-portrait-word.png')
C2             = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-42-portrait-word.png')
C3            = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-159-portrait-word.png')
C4            = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-258-portrait-word.png')


#s, erg, BL, _, _ = trainPNG.IMOP.getColumnsCoordinates(C1, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)

#a1,b1            = trainPNG.IMOP.getColMinMaxCC(2, 2, C1, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)
#a2,b2            = trainPNG.IMOP.getColMinMaxCC(2, 1, C1, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)

#l1 = trainPNG.IMOP.calcSplitJPGs(Image.fromarray(C1), 2)
#C1[:, l1[0][0]] = 0
#C1[:, l1[0][1]-1] = 0
#C1[:, l1[1][0]] = 0
#C1[:, l1[1][1]-1] = 0

#dL = trainPNG.IMOP.getColumns(C1, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)
CC1 = trainPNG.IMOP.makeCopiesOfColumns(C4, 1, 3, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)
CC2 = trainPNG.IMOP.makeCopiesOfColumns(C4, 2, 3, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)
CC3 = trainPNG.IMOP.makeCopiesOfColumns(C4, 1, 2, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)




#Image.fromarray(C1).show()


l2 = trainPNG.IMOP.calcSplitJPGs(Image.fromarray(C2), 1)
#if len(l2) ==0:
  # Image.fromarray(C2).show()


l3 = trainPNG.IMOP.calcSplitJPGs(Image.fromarray(C3), 2)
C3[:, l3[0][0]] = 0
C3[:, l3[0][1]-1] = 0
C3[:, l3[1][0]] = 0
C3[:, l3[1][1]-1] = 0

#Image.fromarray(C3).show()




l4 = trainPNG.IMOP.calcSplitJPGs(Image.fromarray(C4), 3)
C4[:, l4[0][0]] = 0
C4[:, l4[0][1]-1] = 0
C4[:, l4[1][0]] = 0
C4[:, l4[1][1]-1] = 0
C4[:, l4[2][0]] = 0
C4[:, l4[2][1]-1] = 0

#Image.fromarray(C4).show()





SQL = "select namePDFDocument, page, numberOfColumns from TAO where namePDFDocument like '%%train/train%%' order by page"
rs             = trainPNG.con.execute(SQL)
COLN           = list(rs.keys())
LLL            = list(rs)

ERG = []
R = tqdm(1*LLL)
for l in R:
   A, page, numberOfColumns = l
   C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-' + str(page) + '-portrait-word.png')
   s, erg, BL, _, _ = trainPNG.IMOP.getColumnsCoordinates(C, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)
   ERG.append([page, numberOfColumns, len(s), s])

trainPNG.con.close()


E  = list(filter( lambda x: x[1] != x[2] , ERG))
pl = list(map(lambda x: x[0], E))
ML = []
for p in 1*pl:
   C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-' + str(p) + '-portrait-word.png')   
   ML.append(C)

print("Fehler-Quote = " + str(round(len(E)/len(LLL),4)))

#CL = list(map(lambda x: trainPNG.IMOP.getColumns2( x, white=255, windowSize=200, stepSize= 50, bound=0.99, debug=False), ML))
#[s, erg, BL, start, ende] = CL[2]
#xmms, start, ende, mm, b =trainPNG.IMOP.getColumnsWindow(BL[2], start, ende)

#ergt = list(map(lambda x: trainPNG.IMOP.getColumnsWindow(x, start, ende), BL))


C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-' + str(855) + '-portrait-word.png')   
s, erg, BL, start, ende = trainPNG.IMOP.getColumnsCoordinates( C, white=255, windowSize=450, stepSize= 50, bound=0.99, part=8, debug=False)


nn =0
xmms, start, ende, mm, b =trainPNG.IMOP.getColumnsWindow(BL[nn], start, ende)
#L = [[43, 71], [72, 98], [134, 170], [171, 292], [294, 500], [550, 600]]
#gg=trainPNG.IMOP.unitedIntervals(L)



B = BL[nn].copy()
l = list(np.concatenate(xmms))
for a in l:
   B[:, a] = 0

Image.fromarray(B).show()




#***
#*** MAIN PART
#***
#
#  exec(open("testGeneral.py").read())
#


# 115, 518, 519, 664, 666, 717, 855, 880

"""

