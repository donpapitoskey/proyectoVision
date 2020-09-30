#Importar aquí las librerías a utilizar

from PyQt5 import uic, QtWidgets
from interfaz import *
import sys
import cv2 
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from scipy import signal
import numpy as np
#from matplotlib import pyplot as plt

class Window2(QMainWindow):                           # <===
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window22222")
        self.resize(650, 505)
        self.newCentralWidget = QtWidgets.QWidget(self)
        self.groupBox_3 = QtWidgets.QGroupBox(self.newCentralWidget)
        self.groupBox_3.setGeometry(0,0,650,505)
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_3)
        
        

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def window2(self):                                             # <===
        self.w = Window2()
        self.w.show()
        self.hide()
    
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.img=[]
        self.img2=[]
        self.btnSelecArchivo.clicked.connect(self.abrirImg)
        self.btnGlobal.clicked.connect(self.histogramaGlobal)
        self.btnHorizontal.clicked.connect(self.histogramaHor)
        self.btnVertical.clicked.connect(self.histogramaVer)
        self.btnFunc.clicked.connect(self.histogramaEcualizado)
        self.btnFunc1.clicked.connect(self.aplicarClahe)
        self.btnFunc2.clicked.connect(self.highBoost)
        x=np.array(np.arange(0,256,1)).reshape((1,256))
        xMat=np.repeat(x,256, axis=0)
        yMat=xMat.transpose()
        self.test = np.divide(np.add(xMat,yMat),2)
        #Aquí van los botones
        
    #Aquí van las nuevas funciones
    def abrirImg(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:')
        
        
        #self.wIm = Window2()
        #self.wIm.imageContainer = QtWidgets.QGraphicsView(self.wIm.groupBox_3)
        #self.wIm.gridLayout.addWidget(self.wIm.imageContainer, 0, 1, 1, 1)
        #self.wIm.setCentralWidget(self.wIm.newCentralWidget)
        #self.wIm.setWindowTitle("Image Container")
        #self.wIm.show()

        if filePath != "":
            print ("Dirección",filePath) #Opcional imprimir la dirección del archivo
            self.img=cv2.imread(filePath,)
            self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            img = pg.ImageItem()
            img.setImage(self.img)
            #self.plotImg.addItem(img)
            #PIXMAP= QPixmap(filePath)
            #PIXMAP=PIXMAP.scaled(self,self.label_2.width(), self.label_2.height())
            #self.label_2.setPixmap(PIXMAP)
            
            
    def histogramaGlobal(self):
        self.plotHisto.clear()

        self.wGlob = Window2()
        self.wGlob.plotHisto = PlotWidget(self.wGlob.groupBox_3)
        self.wGlob.gridLayout.addWidget(self.wGlob.plotHisto, 0, 1, 1, 1)
        self.wGlob.setCentralWidget(self.wGlob.newCentralWidget)
        self.wGlob.setWindowTitle("Global")
        self.wGlob.show()
        self.wGlob.plotHisto.clear()
        #img2=self.img
        img2=cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
        #hist=hist.T
        #print(len(hist))
        #print(type(hist))
        x = np.linspace(0,255,256)
        #x=x.T
        #print(type(x))
        hist=hist.reshape(-1)
        
        self.plotHisto.plot(x,hist, title=('Histograma Global')) 
        self.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
        self.wGlob.plotHisto.plot(x,hist, title=('Histograma Global')) 
        self.wGlob.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.wGlob.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
        
        
       
        #pg.plot(x,hist, title=('Histograma Global') ) 
        #plt.xlabel('intensidad de iluminacion')
        
       
        
    
    def histogramaHor(self):
        self.wHor = Window2()
        self.wHor.plotHisto = PlotWidget(self.wHor.groupBox_3)
        self.wHor.gridLayout.addWidget(self.wHor.plotHisto, 0, 1, 1, 1)
        self.wHor.setCentralWidget(self.wHor.newCentralWidget)
        self.wHor.setWindowTitle("Horizontal")
        self.wHor.show()
        self.wHor.plotHisto.clear()

        
        self.plotHisto.clear()
        #imag3=self.img
        # Conversion a escala de grices
        imag3 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        yhist=np.mean(imag3,axis=1)
        y = -np.linspace(0,len(yhist),len(yhist))
        self.plotHisto.plot(yhist,y, title=('Histograma Horizontal')) 
        self.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.plotHisto.setLabel('bottom','Intensidad de iluminacion' )        

        self.wHor.plotHisto.plot(yhist,y, title=('Histograma Horizontal')) 
        self.wHor.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.wHor.plotHisto.setLabel('bottom','Intensidad de iluminacion' )        
                  
    def histogramaVer(self):

        self.wVer = Window2()
        self.wVer.plotHisto = PlotWidget(self.wVer.groupBox_3)
        self.wVer.gridLayout.addWidget(self.wVer.plotHisto, 0, 1, 1, 1)
        self.wVer.setCentralWidget(self.wVer.newCentralWidget)
        self.wVer.setWindowTitle("Horizontal")
        self.wVer.show()
        self.wVer.plotHisto.clear()
        self.plotHisto.clear()

        #imag4=self.img
        imag4 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        xhist=np.mean(imag4,axis=0)
        self.plotHisto.plot(xhist, title=('Histograma Vertical'))         
        self.plotHisto.setLabel('left','Intensidad de iluminacion' )
        self.plotHisto.setLabel('bottom','Cantidad de pixeles' )
        self.wVer.plotHisto.plot(xhist, title=('Histograma Vertical'))         
        self.wVer.plotHisto.setLabel('left','Intensidad de iluminacion' )
        self.wVer.plotHisto.setLabel('bottom','Cantidad de pixeles' )

    def function1(self):
        
        
        pg.image(self.test)
        pg.image(np.subtract(255,self.img))
    
    def aplicarClahe(self):

        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
        lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        #pg.image(l)
        pg.image(self.img)
        hist = cv2.calcHist([l], [0], None, [256], [0, 256])
        #plt.plot(hist, color='gray' )
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        #pg.image(l2)
        hist = cv2.calcHist([l2], [0], None, [256], [0, 256])
        #plt.plot(hist, color='gray' )
        lab = cv2.merge((l2,a,b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to BGR
        pg.image( img2)
    
    def highBoost(self):
        eg_1 = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        h_channel,s_channel,v_channel = cv2.split(eg_1)

        #Creando la mascara
        kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])

        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

        #Convolusion de mascara e imagen
        img_rst = cv2.filter2D(v_channel,-1,kernel)
        img_rst = cv2.merge((h_channel, s_channel, img_rst))
        self.highboost = cv2.cvtColor(img_rst,cv2.COLOR_HSV2RGB)
        pg.image(self.highboost)

    def histogramaEcualizado(self):
        eg_1 = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        h_channel,s_channel,v_channel = cv2.split(eg_1)
        eh = cv2.equalizeHist(v_channel)
        h_eHSV = cv2.calcHist([eh],[0],None,[256],[0,256])
        h_eHSV=h_eHSV.reshape(-1)
        self.wEcu = Window2()
        self.wEcu.plotHisto = PlotWidget(self.wEcu.groupBox_3)
        self.wEcu.gridLayout.addWidget(self.wEcu.plotHisto, 0, 1, 1, 1)
        self.wEcu.setCentralWidget(self.wEcu.newCentralWidget)
        self.wEcu.setWindowTitle("Ecualizado")
        self.wEcu.show()
        self.wEcu.plotHisto.clear()
        self.wEcu.plotHisto.plot(h_eHSV, title=('Histograma Ecualizado'))         
        self.wEcu.plotHisto.setLabel('left','Intensidad de iluminacion' )
        self.wEcu.plotHisto.setLabel('bottom','Cantidad de pixeles' )

    def normHistogram(self):
        self.wNormGlob = Window2()
        self.wNormGlob.plotHisto = PlotWidget(self.wNormGlob.groupBox_3)
        self.wNormGlob.gridLayout.addWidget(self.wNormGlob.plotHisto, 0, 1, 1, 1)
        self.wNormGlob.setCentralWidget(self.wNormGlob.newCentralWidget)
        self.wNormGlob.setWindowTitle("Global Normalizado")
        self.wNormGlob.show()
        self.wNormGlob.plotHisto.clear()
        #img2=self.img
        img2=cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
        #hist=hist.T
        #print(len(hist))
        #print(type(hist))
        x = np.linspace(0,255,256)
        #x=x.T
        #print(type(x))
        hist=hist.reshape(-1)
        print(hist)

        self.plotHisto.plot(x,hist, title=('Histograma Global Normalizado')) 
        self.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
        self.wGlob.plotHisto.plot(x,hist, title=('Histograma Global')) 
        self.wGlob.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.wGlob.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
        

            
             
  
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    
    
    