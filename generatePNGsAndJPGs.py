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



# ****************************
# *** begin JPG            ***
# ****************************



trainJPG         = dOM.imageGeneratorJPG(pathToPDF       = pathPDFFilename1, 
                                         pdfFilename     = PDFFilename1, 
                                         outputFolder    = pathJPGs1, 
                                         output_file     = 'train', 
                                         pageStart       = 1,
                                         pageEnd         = 0, 
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


lfgbJPG         = dOM.imageGeneratorJPG(pathToPDF        = pathPDFFilename3, 
                                         pdfFilename     = PDFFilename3, 
                                         outputFolder    = pathJPGs3, 
                                         output_file     = 'lf-gb2019finalg-2-columns-pages-with-at-least-one-table', 
                                         pageStart       = 1,
                                         pageEnd         = 0, 
                                         scanedDocument  = False, 
                                         dpi             = 200, 
                                         generateJPGWith = 'tesseract', 
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
    
lfgbJPG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
lfgbJPG.con     = trainJPG.engine.connect()
#lfgbJPG.L       = [113,115]  
#lfgbJPG.generateJPG()


challengeJPG         = dOM.imageGeneratorJPG(pathToPDF   = pathPDFFilename2, 
                                         pdfFilename     = PDFFilename2, 
                                         outputFolder    = pathJPGs2, 
                                         output_file     = 'challenge', 
                                         pageStart       = 1,
                                         pageEnd         = 0, 
                                         scanedDocument  = False, 
                                         dpi             = 200, 
                                         generateJPGWith = 'tesseract', 
                                         windowSize      = 450, 
                                         stepSize        = 50, 
                                         bound           = 0.99, 
                                         part            = 8, 
                                         ub              = 5, 
                                         size            = (595, 842) )
    
challengeJPG.engine  = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
challengeJPG.con     = trainJPG.engine.connect()
#challengeJPG.L       = [113,115]  
#challengeJPG.generateJPG()


# ****************************
# *** end JPG              ***
# ****************************





# ****************************
# *** begin PNG            ***
# ****************************

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
#trainPNG.generatePNG()


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
#lfgbPNG.generatePNG()



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
#challengePNG.generatePNG()


# ****************************
# *** end PNG              ***
# ****************************




#***
#*** MAIN PART
#***
#
#  exec(open("generatePNGsAndJPGs.py").read())
#


# 115, 518, 519, 664, 666, 717, 855, 880



