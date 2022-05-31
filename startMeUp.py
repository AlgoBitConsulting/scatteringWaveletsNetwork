
#from docScatWaveNet import dataOrganisationModule as dOM, misc as MISC, morletModule as MM, scatteringTransformationModule as ST, tableFinder as TF, DFTForSCN as DFT

import socket
import os, numpy as np
import sys, subprocess, os,glob
from PIL import Image, ImageDraw, ImageOps, ImageFont

pi, exp, log, abs, sqrt, fft, mult, mat, tp = np.pi, np.exp, np.log, np.abs, np.sqrt, np.fft.fft, np.multiply, np.matrix, np.transpose
cos,sin = np.cos, np.sin
matmul  = np.matmul
inv     = np.linalg.inv
diag    = np.diag
imag,real = np.imag, np.real


class boxMaster:
   def __init__(self, name='ka'):
      self.name = name  

def deleteFiles(FL, dir):
   L = []
   for fl in FL:
      fname = dir + fl 
      if os.path.exists(fname): 
         os.remove(fname)
      L.append(fname)
   return(L)



workingPath      = os.getcwd() + '/'

sys.path.append(workingPath + 'src/docScatWaveNet/')
import misc as MISC
import scatteringTransformationModule as ST
import dataOrganisationModule as dOM
import morletModule as MM  
import tableFinder as TF


#ss = "python -m pip install --no-index " + workingPath + "dist/docScatWaveNet2-0.0.1-py3-none-any.whl"
#subprocess.check_output(ss, shell=True,executable='/bin/bash')


generatePNGsAndJPgs = False

# *****************************
# *** start init generators ***
# *****************************



np.set_printoptions(suppress=True)


if socket.gethostname() == 'home':
   pathPDFFilename      = '/home/markus/anaconda3/python/pngs/train/'
   pathPNGs             = '/home/markus/anaconda3/python/pngs/train/word/'
   PDFFilename          = 'train'
   pathModels           = '/home/markus/anaconda3/python/data/'
   namePDFL             = ['/home/markus/anaconda3/python/pngs/train/train', '/home/markus/anaconda3/python/pngs/train/lf-gb2019finalg-2-columns-pages-with-at-least-one-table']

else:
   pathModels           = '/home/markus/GIT/scatteringWaveletsNetwork/train/models/'
   PDFFilename          = 'train_hochkant'
   namePDFL             = ['/home/markus/GIT/scatteringWaveletsNetwork/train_hochkant']


trainPNG             = dOM.imageGeneratorPNG(pathToPDF   = pathPDFFilename, 
                                         pdfFilename     = PDFFilename, 
                                         outputFolder    = pathPNGs, 
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
#trainPNG.engine      = create_engine('mysql+pymysql://markus:venTer4hh@localhost/TAO')
#trainPNG.con         = trainPNG.engine.connect()


pathPDFFilename  = workingPath
PDFFilename      = 'challenge'
pathJPGs         = workingPath + 'test/'
pathPNGs         = workingPath + 'challenge/word/'
pathOutput       = workingPath + 'test/'


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
                                             size            = (596, 842) )


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
                                             size            = (596, 842) )

if generatePNGsAndJPgs:
   challengePNG.generate(getNOCfromDB=False)
   challengeJPG.generate(getNOCfromDB=False)

# *****************************
# *** end init generators   ***
# *****************************


white                = 255
black                = 0 


#white                = 255
#black                = 0 

# ****************************
# *** begin load data      ***
# ****************************

#FL         = ['TA-bB-H-PNG-' , 'TA-bB-V-PNG-' ,  'TA-bBHV-H-PNG-',  'TA-bBHV-V-PNG-'  , 'HL-bB-H-PNG-'      , 'HL-bB-V-PNG-'     ,'HL-bBHV-H-PNG-'   ,'HL-bBHV-V-PNG-'   ]
FL         = ['TA-bB-H-JPG-' , 'TA-bB-V-JPG-' ,  'TA-bBHV-H-JPG-',  'TA-bBHV-V-JPG-'  , 'HL-bB-H-JPG-'      , 'HL-bB-V-JPG-'     ,'HL-bBHV-H-JPG-'   ,'HL-bBHV-V-JPG-'   ]

L          = deleteFiles(list(map(lambda x: x[0:-1], FL)), workingPath + "rf/")

try:
   a = len(INFO.DBOX)
except:
   print("loading random forests...")
   DL                = []
   INFO              = TF.makeINFO()
   INFO.path         = workingPath + 'rf/compressed/'
   INFO.pathRF       = workingPath + 'rf/'
   INFO.kindOfImages = 'JPG'  
   INFO.white        = 255
   INFO.black        = 0    
   INFO.copyHL       = True
   INFO.flatten      = True
   INFO.KBOX         = ['TA']
   INFO.MBOX         = ['bB', 'bBHV']
   INFO.DBOX         = ['H', 'V']

   for kindOfBox in INFO.KBOX:
      for method in INFO.MBOX:
         for direction in INFO.DBOX:
            fname = kindOfBox + '-' + method + '-' + direction + '-' + INFO.kindOfImages

            ss = "unzstd " + INFO.path + fname + ".zst -o "+ INFO.pathRF + fname
            subprocess.check_output(ss, shell=True, executable='/bin/bash')

            DATA = MISC.loadIt(INFO.pathRF + fname)
            if DATA.INFO.kindOfImages != INFO.kindOfImages:
               print("Warning: INFO.kindOfImages = " + INFO.kindOfImages + " is not the same as DATA.INFO.kindOfImages = " + DATA.INFO.kindOfImages + " from " + fname)
            O    = getattr( getattr(INFO,  kindOfBox), method)
            setattr(O, direction,  getattr( getattr( getattr(DATA.INFO,  kindOfBox), method), direction)) 
            S    = getattr(O, direction)
            setattr(S, 'rf', DATA.rf)
            setattr(S, 'windowSize', getattr( getattr( getattr( getattr(DATA.INFO,  kindOfBox), method), direction), 'windowSize'))
            setattr(S, 'stepSize', getattr( getattr( getattr( getattr(DATA.INFO,  kindOfBox), method), direction), 'stepSize'))

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
# *** start INFO           ***
# ****************************

INFO.TA.bB.V.windowSize = 20
INFO.TA.bBHV.V.windowSize = 20

print("bB-H stepSize/windowSize:   " + str(INFO.TA.bB.H.stepSize ) + "/" + str(INFO.TA.bB.H.windowSize ) )
print("bB-V stepSize/windowSize:   " + str(INFO.TA.bB.V.stepSize ) + "/" + str(INFO.TA.bB.V.windowSize ) )
print("bBHV-H stepSize/windowSize:   " + str(INFO.TA.bBHV.H.stepSize ) + "/" + str(INFO.TA.bBHV.H.windowSize ) )
print("bBHV-V stepSize/windowSize:   " + str(INFO.TA.bBHV.V.stepSize ) + "/" + str(INFO.TA.bBHV.V.windowSize ) )


setattr(INFO.TA, 'correction-H', 0.30)  #0.35
setattr(INFO.TA, 'correction-V', 0.15)
setattr(INFO.TA, 'weightbBHV-V', 0.5)
setattr(INFO.TA, 'weightbB-V'  , 0.5)
setattr(INFO.TA, 'weightbBHV-H', 0.5)
setattr(INFO.TA, 'weightbB-H'  , 0.5)

#setattr(INFO.HL, 'correction-H', 0.1)
#setattr(INFO.HL, 'correction-V', 0.2)
#setattr(INFO.HL, 'weightbBHV-V', 0.5)
#setattr(INFO.HL, 'weightbB-V'  , 0.5)
#setattr(INFO.HL, 'weightbBHV-H', 0.5)
#setattr(INFO.HL, 'weightbB-H'  , 0.5)


# ****************************
# *** end INFO             ***
# ****************************


# ****************************
# *** begin init SWC       ***
# ****************************

adaptMatrixCoef      = INFO.TA.bB.H.adaptMatrixCoef

MAT                  = dOM.matrixGenerator('downsampling')
MAT.description      = "TEST"
C1                   = MAT.generateMatrixFromImage(workingPath + 'C1.png')
Ct                   = MAT.downSampling(C1, adaptMatrixCoef)                
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

# *****************************
# *** main                  ***
# *****************************

#***
#*** MAIN PART
#***
#
#  exec(open("startMeUp.py").read())
#
##   

try:
   a = len(BIGINFO)
except:
   print("creating BIGINFO...")
   BIGINFO = {}


INFO.kindOfImagesCustomer = 'JPG'

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


INFO.MAT         = MAT
INFO.DATA        = DATA
INFO.columns     = columns
INFO.STPE        = STPE

###########################################################################

def findStartAndEnd(erg, WL,lenErg=3, stin=0, enin=1):

      foundStart = False
      foundEnd   = False
      start      = 0
      end        = 0
      ii         = 0
      jj         = 0
      boxL       = []
      
      windowSize = WL[0][1][1] - WL[0][1][0]
      stepSize   = WL[1][1][1] - WL[0][1][1]
     
      while ii < len(WL):
         if erg[ii] == 1:
            W          = WL[ii]
            start      = W[1][0]                #+ 0*int(0.5*(windowSize- 2*stepSize)) 
            foundStart = True
                 
         if foundStart:
            while ii < len(WL) and not(foundEnd):
               if np.sum( erg[ii:ii+lenErg]) == 0: 
                  W          = WL[ii]
                  end        = W[1][1]                 #- 0*int(0.5*(windowSize- 2*stepSize)) 
                  foundEnd   = True
                  boxL.append([start, end])
                  
               else:
                  ii = ii+1       
         if foundEnd:
            foundStart = False
            foundEnd   = False  

         ii = ii+1
         
      return(boxL)

###########################################################################


### Let's start

page          = 28
INFO.page     = page
RESULTS       = TF.pageTablesAndCols(page=page, generator=trainPNG, BIGINFO = BIGINFO, INFO=INFO, generateImageOTF=generateImageOTF, calcSWCs=calcSWCs, withScalePlot=withScalePlot)
RESULTS.img_TA.show()


#img, TCL      = TF.getResults(page, challengeJPG, RESULTS.KL, RESULTS.MIDL3, RESULTS.BOXL, RESULTS.rLN_TAt)
#img.show()
#try:
#   print(TCL[0])
#except:
#   print("no tables found")

#import importlib
#importlib.reload('src/docScatWaveNet/tableFinder.py')

OM             = getattr(INFO, kindOfBox)
OH             = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'H')  
OV             = getattr( getattr( getattr(INFO, kindOfBox), INFO.method), 'V')  
col            = 0
M_H, M_V       = getattr(OM, 'MH'+ str(col)), getattr( OM, 'MV'+ str(col))
WLH, WLV       = getattr(OH, 'WL'+ str(col)).WL, getattr(OV, 'WL'+ str(col)).WL
boxL_H, boxL_V = TF.findStartAndEnd(M_H[:, 3], WLH, 3), TF.findStartAndEnd(M_V[:, 3], WLV, 3)

#boxL_H, boxL_V = [], []

#for ii in range(len(M_H)):
#   if M_H[ii][2]   


rLt, rL, rLn   = TF.getBoxes(boxL_H, boxL_V)    




