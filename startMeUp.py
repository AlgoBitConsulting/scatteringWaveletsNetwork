
### standardmäßig in python installiert
import sys, subprocess
from os import system
from PIL import Image, ImageDraw, ImageOps, ImageFont


### eigene Module
"""
sys.path.append('/home/markus/python/scatteringWaveletsNetworks/modules')
sys.path.append('/home/markus/anaconda3/python/development/modules')
import misc as MISC
import scatteringTransformationModule as ST
import dataOrganisationModule as dOM
import morletModule as MM  
import tableFinder as TF
"""
from docScatWaveNet import dataOrganisationModule as dOM, misc as MISC, morletModule as MM, scatteringTransformationModule as ST, tableFinder as TF


### zu installierende Module
from tqdm import tqdm
import numpy as np


############################################################################# 

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real



###################################################################################################################



class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  

#***
#*** MAIN PART
#***
#
#  exec(open("startMeUp.py").read())
#

generatePNGsAndJPgs = False


# *****************************
# *** start init generators ***
# *****************************

np.set_printoptions(suppress=True)

pathPDFFilename = '/home/markus/anaconda3/python/development/'
PDFFilename     = 'challenge'
pathJPGs        = '/home/markus/anaconda3/python/development/test/'
pathPNGs        = '/home/markus/anaconda3/python/development/challenge/word/'
pathOutput      = '/home/markus/anaconda3/python/development/test/'


challengePNG         = dOM.imageGeneratorPNG(pathToPDF       = pathPDFFilename, 
                                             pdfFilename     = PDFFilename, 
                                             outputFolder    = pathPNGs, 
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


challengeJPG         = dOM.imageGeneratorJPG(pathToPDF       = pathPDFFilename, 
                                             pdfFilename     = PDFFilename, 
                                             outputFolder    = pathPNGs,  
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

if generatePNGsAndJPgs:
   challengePNG.generate(getNOCfromDB=False)
   challengeJPG.generate(getNOCfromDB=False)

# *****************************
# *** end init generators   ***
# *****************************


# ****************************
# *** begin init SWC       ***
# ****************************


MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = MAT.generateMatrixFromImage('/home/markus/anaconda3/python/development/C1.png')
Ct                   = MAT.downSampling(C1, 3)                
dx,dy                = 0.15, 0.15
SWO_2D               = MM.SWO_2D(Ct, round(Ct.shape[1]*0.5*dx,3), round(Ct.shape[0]*0.5*dy,3))
SWO_2D.init_eta      = 2
SWO_2D.kk            = 1
SWO_2D.sigma         = mat([[SWO_2D.kk,0],[0,SWO_2D.kk]])
SWO_2D.J             = 0   
SWO_2D.nang          = 8
SWO_2D.ll            = 2
SWO_2D.jmax          = SWO_2D.J
SWO_2D.m             = 2   
SWO_2D.outer         = False
SWO_2D.onlyCoef      = True
SWO_2D.allLevels     = False
SWO_2D.normalization = False    # (m=2 wird mit m=1-Wert normalisiert)

# ****************************
# *** end init SWC         ***
# ****************************

white                = 255
black                = 0 

# ****************************
# *** begin load data      ***
# ****************************

ss = "rm rf/*"
try:
   subprocess.check_output(ss, shell=True,executable='/bin/bash')
except:
   print("Problems removing rfs...")

DL = ['-18-04-2022-15:47:03','-18-04-2022-16:12:09','-18-04-2022-16:45:52','-18-04-2022-17:13:58', '-18-04-2022-17:42:09','-18-04-2022-18:06:56','-18-04-2022-18:40:05','-18-04-2022-19:07:48']

try:
   a = len(FL)
except:
   print("loading random forests...")
   INFO              = TF.makeINFO()
   INFO.path         = '/home/markus/anaconda3/python/development/rf/compressed/'
   INFO.pathRF       = '/home/markus/anaconda3/python/development/rf/'
   INFO.kindOfImages = 'PNG'  
   INFO.white        = 255
   INFO.black        = 0    
   INFO.copyHL       = True
   INFO.flatten      = True
   FL                = ['TA-bB-H-PNG-' , 'TA-bB-V-PNG-' ,  'TA-bBHV-H-PNG-',  'TA-bBHV-V-PNG-'  , 'HL-bB-H-PNG-'      , 'HL-bB-V-PNG-'     ,'HL-bBHV-H-PNG-'   ,'HL-bBHV-V-PNG-'   ]
   INFO.KBOX         = ['TA','HL']
   INFO.MBOX         = ['bB', 'bBHV']
   INFO.DBOX         = ['H', 'V']

   for kindOfBox in INFO.KBOX:
      for method in INFO.MBOX:
         for direction in INFO.DBOX:
            fname = kindOfBox + '-' + method + '-' + direction + '-' + INFO.kindOfImages

            ss = "unzstd " + INFO.path + fname + ".zst --output-dir-flat="+ INFO.pathRF
            subprocess.check_output(ss, shell=True,executable='/bin/bash')

            DATA = MISC.loadIt(INFO.pathRF + fname)
            if DATA.INFO.kindOfImages != INFO.kindOfImages:
               print("Warning: INFO.kindOfImages = " + INFO.kindOfImages + " is not the same as DATA.INFO.kindOfImages = " + DATA.INFO.kindOfImages + " from " + fname)
            O    = getattr( getattr(INFO,  kindOfBox), method)
            setattr(O, direction,  getattr( getattr( getattr(DATA.INFO,  kindOfBox), method), direction)) 
            S    = getattr(O, direction)
            setattr(S, 'rf', DATA.rf)
            try:
               INFO.onlyWhiteBlack  = DATA.INFO.onlyWhiteBlack
               INFO.wBB             = DATA.INFO.wBB
            except:
               print("INFO contains no information about onlyBlackWhite option...seting values to default!")
               INFO.onlyWhiteBlack = False
               INFO.wBB            = 180

   print("...loading random forests done")

# ****************************
# *** end load data        ***
# ****************************


# ****************************
# *** start init INFO      ***
# ****************************

stepSize_H                = 5
windowSize_H              = 30
INFO.TA.bB.H.stepSize     = stepSize_H 
INFO.TA.bB.H.windowSize   = windowSize_H 
INFO.TA.bBHV.H.stepSize   = stepSize_H 
INFO.TA.bBHV.H.windowSize = windowSize_H 
 
windowSize_V              = 80
stepSize_V                = 5
INFO.TA.bB.V.windowSize   = windowSize_V 
INFO.TA.bB.V.stepSize     = stepSize_V 
INFO.TA.bBHV.V.windowSize = windowSize_V  
INFO.TA.bBHV.V.stepSize   = stepSize_V  


setattr(INFO.TA, 'correction-H', 0.35)  #0.35
setattr(INFO.TA, 'correction-V', 0.15)
setattr(INFO.TA, 'weightbBHV-V', 0.5)
setattr(INFO.TA, 'weightbB-V'  , 0.5)
setattr(INFO.TA, 'weightbBHV-H', 0.5)
setattr(INFO.TA, 'weightbB-H'  , 0.5)

setattr(INFO.HL, 'correction-H', 0.1)
setattr(INFO.HL, 'correction-V', 0.2)
setattr(INFO.HL, 'weightbBHV-V', 0.5)
setattr(INFO.HL, 'weightbB-V'  , 0.5)
setattr(INFO.HL, 'weightbBHV-H', 0.5)
setattr(INFO.HL, 'weightbB-H'  , 0.5)


# ****************************
# *** end init INFO        ***
# ****************************


# *****************************
# *** main                  ***
# *****************************

try:
   a = len(BIGINFO)
except:
   print("creating BIGINFO...")
   BIGINFO = {}


INFO.kindOfImagesCustomer = 'PNG'

if INFO.kindOfImagesCustomer != INFO.kindOfImages:
      print("Warning: kind of images=" + INFO.kindOfImagesCustomer+ " for fitting rf is not the same as the kind of images=" + INFO.kindOfImages+ " for prediction of images!")

if INFO.kindOfImagesCustomer == 'PNG':   
   STPE             = dOM.stripe('.png', C1, stepSize=0, windowSize=0, direction='H', SWO_2D=SWO_2D)
   generator        = challengePNG

if INFO.kindOfImagesCustomer == 'JPG':
   STPE             = dOM.stripe('.jpg', C1, stepSize=0, windowSize=0, direction='H', SWO_2D=SWO_2D)
   generator        = challengeJPG


columns          = dOM.columns()
ss               = input("calculate SWCs (Y/N) ?")
calcSWCs         = False
if ss=='Y':
   calcSWCs = True

STPE.dd          = 0.20
STPE.tol         = 30
generateImageOTF = False
withScalePlot    = True
page             = 2

INFO.MAT         = MAT
INFO.DATA        = DATA
INFO.columns     = columns
INFO.STPE        = STPE
INFO.page        = page

### Let's start


RESULTS       = TF.pageTablesAndCols(page=page, generator=generator, BIGINFO = BIGINFO, INFO=INFO, generateImageOTF=generateImageOTF, calcSWCs=calcSWCs, withScalePlot=withScalePlot)
RESULTS.img_TA.show()

IMGL, TAB     = [], []
for ii in range(len(RESULTS.KL)):
   tableNumber   = ii
   img, col      = TF.getResults(page, tableNumber, challengeJPG, RESULTS.KL, RESULTS.MIDL3, RESULTS.BOXL)
   IMGL.append(img)
   TAB.append(col)



