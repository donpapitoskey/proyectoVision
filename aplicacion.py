#Importar aquí las librerías a utilizar

from PyQt5 import uic, QtWidgets
from interfaz import *
import sys
import cv2 
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QPixmap
import numpy as np
#from matplotlib import pyplot as plt


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.img=[]
        self.btnSelecArchivo.clicked.connect(self.abrirImg)
        self.btnGlobal.clicked.connect(self.histogramaGlobal)
        self.btnHorizontal.clicked.connect(self.histogramaHor)
        self.btnVertical.clicked.connect(self.histogramaVer)
        
        
        #Aquí van los botones
        
    #Aquí van las nuevas funciones
    def abrirImg(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:')
        if filePath != "":
            print ("Dirección",filePath) #Opcional imprimir la dirección del archivo
            self.img=cv2.imread(filePath)
            img = pg.ImageItem()
            img.setImage(self.img)
            #self.plotImg.addItem(img)
            PIXMAP= QPixmap(filePath)
            #PIXMAP=PIXMAP.scaled(self,self.label_2.width(), self.label_2.height())
            self.label_2.setPixmap(PIXMAP)
            
            
    def histogramaGlobal(self):
        self.plotHisto.clear()
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
        
        
       
        #pg.plot(x,hist, title=('Histograma Global') ) 
        #plt.xlabel('intensidad de iluminacion')
        
       
        
    
    def histogramaHor(self):
        self.plotHisto.clear()
        #imag3=self.img
        # Conversion a escala de grices
        imag3 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        # Calculo de Histograma Horizontal
        x_sum = cv2.reduce(imag3, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        x_sum_len = len(x_sum[0])
        x_sum_norm = x_sum[0]/(x_sum_len)
        x_sum_norm = x_sum_norm.round(0)
        x_sum_norm = x_sum_norm.astype(int)
        print(x_sum_norm)

        x_sum_x = np.linspace(0,255,256)
        x_sum_x = x_sum_x.astype(int)

        x_sum_y = np.zeros(256)
         
        for i in range(256):
           for j in range(x_sum_len):
               if(x_sum_norm[j]==i):
                  x_sum_y[i] += 1
                  
        #pg.plot(x_sum_x, x_sum_y, title=('Histograma Horizontal')) 
        self.plotHisto.plot(x_sum_x,x_sum_y, title=('Histograma Horizontal')) 
        self.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.plotHisto.setLabel('bottom','Intensidad de iluminacion' )        
                  
    def histogramaVer(self):
        self.plotHisto.clear()
        #imag4=self.img
        imag4 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        y_sum = cv2.reduce(imag4, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        y_sum_len = len(y_sum[:,0])
        y_sum_norm = y_sum[:,0]/(y_sum_len)
        y_sum_norm = y_sum_norm.round(0)
        y_sum_norm = y_sum_norm.astype(int)
        
        x_sum_x = np.linspace(0,255,256)
        x_sum_x = x_sum_x.astype(int)

        y_sum_y = np.zeros(256)
        for i in range(256):
          for j in range(y_sum_len):
            if(y_sum_norm[j]==i):
              y_sum_y[i] += 1
        
        #pg.plot(x_sum_x, y_sum_y, title=('Histograma Vertical')) 
        self.plotHisto.plot(x_sum_x,y_sum_y, title=('Histograma Vertical'))         
        self.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
                  

            
             
  
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    
    
    