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
pathPDFFilename1 = '/home/markus/anaconda3/python/pngs/train/'
PDFFilename1     = 'train'
pathJPGs1        = '/home/markus/anaconda3/python/pngs/train/test/'
pathPNGs1        = '/home/markus/anaconda3/python/pngs/train/word/'
pathOutput1      = '/home/markus/anaconda3/python/pngs/train/test/'


pathPDFFilename2 = '/home/markus/anaconda3/python/pngs/challenge/'
PDFFilename2     = 'challenge'
pathJPGs2        = '/home/markus/anaconda3/python/pngs/challenge/test/'
pathPNGs2        = '/home/markus/anaconda3/python/pngs/challenge/word/'
pathOutput2      = '/home/markus/anaconda3/python/pngs/challenge/test/'


pathPDFFilename3 = '/home/markus/anaconda3/python/pngs/train/'
PDFFilename3     = 'lf-gb2019finalg-2-columns-pages-with-at-least-one-table'
pathPNGs3        = '/home/markus/anaconda3/python/pngs/train/word/'
pathOutput3      = '/home/markus/anaconda3/python/pngs/train/test/'


PDF              = dOM.PDF(pathPDFFilename1, PDFFilename1, pathOutput1)
MAT              = dOM.matrixGenerator('downsampling')
MAT.description  = "TEST"
C1               = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-216-portrait-word.png')


trainJPG         = dOM.imageGeneratorJPG(pathToPDF       = pathPDFFilename1, 
                                         pdfFilename     = PDFFilename1, 
                                         outputFolder    = pathJPGs1, 
                                         output_file     = 'train', 
                                         pageStart       = 1,
                                         pageEnd         = 5, 
                                         scanedDocument  = False, 
                                         dpi             = 200, 
                                         generateJPGWith = 'tesseract', 
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
    
trainJPG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
trainJPG.con     = trainJPG.engine.connect()
#trainJPG.L       = [113,115]  
#trainJPG.generateJPG()


trainPNG         = dOM.imageGeneratorPNG(pathToPDF       = pathPDFFilename1, 
                                         pdfFilename     = PDFFilename1, 
                                         outputFolder    = pathPNGs1, 
                                         output_file     = 'train', 
                                         pageStart       = 1,
                                         pageEnd         = 0,  
                                         scanedDocument  = False,
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
trainPNG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
trainPNG.con     = trainPNG.engine.connect()
trainPNG.generatePNG()


lfgbPNG          = dOM.imageGeneratorPNG(pathToPDF       = pathPDFFilename3, 
                                         pdfFilename     = PDFFilename3, 
                                         outputFolder    = pathPNGs3, 
                                         output_file     = 'lf-gb2019finalg-2-columns-pages-with-at-least-one-table', 
                                         pageStart       = 1,
                                         pageEnd         = 0,  
                                         scanedDocument  = False,
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
lfgbPNG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
lfgbPNG.con     = trainPNG.engine.connect()
lfgbPNG.generatePNG()



challengePNG         = dOM.imageGeneratorPNG(pathToPDF   = pathPDFFilename2, 
                                         pdfFilename     = PDFFilename2, 
                                         outputFolder    = pathPNGs2, 
                                         output_file     = 'challenge', 
                                         pageStart       = 1,
                                         pageEnd         = 0,  
                                         scanedDocument  = False,
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
challengePNG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
challengePNG.con     = challengePNG.engine.connect()
challengePNG.generatePNG()



"""
C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/lf-gb2019finalg-2-columns-pages-with-at-least-one-table-' + str(20) + '-portrait-word.png')
s, erg, BL, start, ende  = challengePNG.IMOP.getColumnsCoordinates(C)
dL                       = challengePNG.IMOP.getColumns(C) 
xmms, start, ende, mm, b =challengePNG.IMOP.getColumnsWindow(BL[0], start, ende)

B = BL[0].copy()
l = list(np.concatenate(xmms))
for a in l:
   B[:, a] = 0

#Image.fromarray(B).show()


#C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/challenge/word/challenge-' + str(6) + '-portrait-word.png')
C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/lf-gb2019finalg-2-columns-pages-with-at-least-one-table-' + str(15) + '-portrait-word.png')
xmm = trainPNG.IMOP.calcSplit(C,2)
xmm2 = trainPNG.IMOP.getColumnsCoordinates(C)[0]




SQL1 = "select namePDFDocument, page, numberOfColumns from TAO where namePDFDocument like '%%challenge%%' order by page"
SQL2 = "select namePDFDocument, page, numberOfColumns from TAO where namePDFDocument like '%%train/train%%' order by page"
rs             = challengePNG.con.execute(SQL1)
COLN           = list(rs.keys())
LLL            = list(rs)

ERG = []
R = tqdm(0*LLL)
for l in R:
   A, page, numberOfColumns = l
   C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/challenge/word/challenge-' + str(page) + '-portrait-word.png')
   s, erg, BL, _, _ = challengePNG.IMOP.getColumnsCoordinates(C)
 
   #C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-' + str(page) + '-portrait-word.png')
   #s, erg, BL, _, _ = trainPNG.IMOP.getColumnsCoordinates(C)

   ERG.append([page, numberOfColumns, len(s), s])

challengePNG.con.close()

E  = list(filter( lambda x: x[1] != x[2] , ERG))
print("Fehler-Quote = " + str(round(len(E)/len(LLL),4)))


pl = list(map(lambda x: x[0], E))
ML = []
for p in 0*pl:
   C = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/pngs/train/word/train-' + str(p) + '-portrait-word.png')   
   ML.append(C)





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
