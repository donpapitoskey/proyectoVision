#Importar aquí las librerías a utilizar

from PyQt5 import uic, QtWidgets
from interfaz import *
import sys
import cv2 
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
import numpy as np
from skimage import morphology
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
        self.btnFunc.clicked.connect(self.histogramaGlobal)
        self.btnFunc1.clicked.connect(self.aplicarClahe)
        self.btnFunc2.clicked.connect(self.highBoost)
        self.btnSegment.clicked.connect(self.recortar2)
        self.btnButterFreq.clicked.connect(self.butterFilterFreq)
        self.btnLaplacian.clicked.connect(self.laplaciano)
        self.btnContorno.clicked.connect(self.contorno)
        self.btnMoments.clicked.connect(self.momentos)
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
        #imag3=self.img
        # Conversion a escala de grices
        imag3 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        yhist=np.mean(imag3,axis=1)
        y = -np.linspace(0,len(yhist),len(yhist))
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
        #imag4=self.img
        imag4 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        xhist=np.mean(imag4,axis=0)
        self.wVer.plotHisto.plot(xhist, title=('Histograma Vertical'))         
        self.wVer.plotHisto.setLabel('left','Intensidad de iluminacion' )
        self.wVer.plotHisto.setLabel('bottom','Cantidad de pixeles' )
    
    def aplicarClahe(self):

        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
        lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        #pg.image(l)
        origi = pg.image(self.img)
        origi.setWindowTitle("Imagen Original")
        hist = cv2.calcHist([l], [0], None, [256], [0, 256])
        #plt.plot(hist, color='gray' )
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        #pg.image(l2)
        hist = cv2.calcHist([l2], [0], None, [256], [0, 256])
        #plt.plot(hist, color='gray' )
        lab = cv2.merge((l2,a,b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to BGR
        modified = pg.image( img2)
        modified.setWindowTitle("CLAHE")
    
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
        origi = pg.image(self.img)
        origi.setWindowTitle("Imagen Original")
        modified = pg.image(self.highboost)
        modified.setWindowTitle("High Boost")

    def histogramaEcualizado(self):
        eg_1 = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        h_channel,s_channel,v_channel = cv2.split(eg_1)
        eh = cv2.equalizeHist(v_channel)
        result = cv2.merge((h_channel,s_channel,eh))
        modified = pg.image(cv2.cvtColor(result,cv2.COLOR_HSV2RGB))
        modified.setWindowTitle("Histograma Ecualizado")
        origi = pg.image(self.img)
        origi.setWindowTitle("Imagen Original")
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

    def butterFilterFreq(self):
        
        # Lectura de Imagenes
        #imagen = cv2.imread('./img/c1anemia-381.jpg')
        imagen_seg = cv2.imread('./img/c1anemia-381bn-ojo-segment.jpg')
        # Conversion a escala de grices

        

        eg_1 = cv2.cvtColor(self.imag,cv2.COLOR_RGB2GRAY)
        eg_2 = cv2.cvtColor(self.imag,cv2.COLOR_RGB2LAB)

        #eg_1 = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        #eg_2 = cv2.cvtColor(self.img,cv2.COLOR_RGB2LAB)
        l_channel,a_channel,b_channel = cv2.split(eg_2)

        Nf = eg_1.shape[0]
        Nc = eg_1.shape[1]

        # Transformada de Fourier 2D
        tf = np.fft.fftshift(np.fft.fft2(eg_1))
        tfAbs = np.abs(tf)

        #Espectro en escala Logaritmica
        tfAbs_log = np.log10(1+tfAbs)
        tfAbs_log_norm = np.uint8(255*tfAbs_log/np.max(tfAbs_log))

        #Diseño del Filtro Paso Alto Butterworth

        F1 = np.arange(-Nf/2+1,Nf/2+1,1)
        F2 = np.arange(-Nc/2+1,Nc/2+1,1)
        [X,Y] = np.meshgrid(F2,F1)

        #Parametros Distancia Euclidiana - Frecuencia de corte - Numero de Orden del filtro
        D = np.sqrt((X**2)+(Y**2))
        D = D/np.max(D)
        Do = 0.0006#0.0007 #Listo
        Do2 = 0.5
        n = 3
        
        Huv2 = 1/(1+np.power(D/Do2,2*n))
        Huv = 1/(1+np.power(D/Do,2*n))              
        Huv = 1 - Huv #Filtro Paso Alto Butterworth
        Huv3 = Huv*Huv2

        #Aplicamos el filtro
        Guv = Huv3*tf
        Guv_abs = np.abs(Guv)
        Guv_abs_log = np.log10(1+Guv_abs)
        Guv_abs_log_norm = np.uint8(255*Guv_abs_log/np.max(Guv_abs_log))

        # IFFT2
        gxy = np.fft.ifft2(Guv)
        gxy = np.abs(gxy)
        #gxy = np.fft.fftshift(gxy)
        gxy = np.uint8(gxy)

        #Imprimir
        result = cv2.merge((gxy,a_channel,b_channel))
        displayable = cv2.cvtColor(result,cv2.COLOR_LAB2RGB)
        origi = pg.image(self.imag)
        origi.setWindowTitle("Imagen Original")
        filtered = pg.image(displayable)
        filtered.setWindowTitle("Imagen Filtrada")

    def laplaciano(self):
        #y,u,v = cv2.split(cv2.cvtColor(self.img,cv2.COLOR_RGB2YUV))
        y,u,v = cv2.split(cv2.cvtColor(self.imag,cv2.COLOR_RGB2YUV))
        
        #calcLaplacian

        #img = cv2.medianBlur(y, 11)                                                # <----- comentar o descomentar para aplicar BLUR
        img = y
        
        #displImage
        N,M = img.shape
        Xvect = np.arange(0,M).reshape((1,M))
        Yvect = np.arange(0,N).reshape((N,1))
        Xmatr = np.repeat(Xvect,N,axis=0)
        Ymatr = np.repeat(Yvect,M,axis=1)
        dispImg = np.multiply(img,np.power(-1,Xmatr+Ymatr))

        fftImg = np.fft.fft2(dispImg)
        #createEuclideanValues
        N,M = img.shape
        Uvect = np.arange(-M/2,M/2).reshape((1,M))
        Vvect = np.arange(-N/2,N/2).reshape((N,1))
        Vmatr = np.repeat(Vvect,M,axis=1)
        Umatr = np.repeat(Uvect,N,axis=0)        
        mask = np.add(np.power(Umatr,2),np.power(Vmatr,2)) 
        
        #createFilter
        maskForLaplacian = 1 + mask
        
        fftImgToInvert = np.multiply(maskForLaplacian,fftImg)

        imageInverted = np.abs(np.fft.ifft2(fftImgToInvert))
 
        imageInverted = np.uint8(255*imageInverted/np.max(imageInverted))
    
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))                     # <----- comentar o descomentar para aplicar CLAHE
        yModified=clahe.apply(imageInverted)                                      # <----- comentar o descomentar para aplicar CLAHE
        

        imageModified = cv2.merge((yModified,u,v))
        #Display results

        
        displayable = cv2.cvtColor(imageModified,cv2.COLOR_YUV2RGB)
        origi = pg.image(self.imag)
        origi.setWindowTitle("Imagen Original")
        filtered = pg.image(displayable)
        filtered.setWindowTitle("Imagen Filtrada")
    def contorno(self):
        #print(self.commonImage)
        #imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        contoured = pg.image(self.commonMask)
        contours, hierarchy = cv2.findContours(self.commonMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # extrae cotorno
        # contour drawing
        img = cv2.drawContours(self.img, contours, -1, (0,255,0), 3) #dibuja los contornos
        #contoured = pg.image(img)
        #contoured.setWindowTitle("Contorno")
        # hulll= and defects calculation
        #cnt = contours[0] # extrae parámetros para hull
        #hull = cv2.convexHull(cnt, returnPoints= False) #calcula los puntos de cambio convexo
        #defects = cv2.convexityDefects(cnt, hull) # calcula los defectos
        #img2 = self.commonImage
        #for i in range(defects.shape[0]): #recorre arreglo de defectos
            #s,e,f,d = defects[i,0] # extrae parámetros de defectos -> inicio (s), final(e), 
            #start = tuple(cnt[s][0]) #ubica el punto de inicio
            #end = tuple(cnt[e][0]) #ubica el punto final
            #far = tuple(cnt[f][0]) #ubica los puntos alejados
            #cv2.line(self.img, start,end, [255,0,255],2) # dibuja los puntos exteriores

        #img2 = cv2.drawContours(self.commonImage, hull, -1, (255,0,255), 3)
        contouredHull = pg.image(img)
        contouredHull.setWindowTitle("Hull")

    def momentos(self):
        # Calculo de canal de luminancia
        eg_2 = cv2.cvtColor(self.commonImage, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(eg_2)

        # Calculo de Histograma Global normalizado
        h_1 = cv2.calcHist([self.commonImage], [0], None, [256], [0, 255])
        h_1 = h_1 / sum(h_1)
        #h_2 = cv2.calcHist([l_channel], [0], None, [256], [0, 255])
        #h_2 = h_2 / sum(h_2)
        # Vector con niveles de gris normalizado
        gris = np.arange(256) / 255
        h_11 = np.zeros(256)
        moments = np.zeros(6)
        for i in range(256):
            h_11[i] = h_1[i][0]

        moments[0] = sum(h_11 * gris)
        # m = moments[1]*np.ones(256)
        # print(h_1)
        for j in range(1, 6):
            moments[j] = sum(((gris - moments[0]) ** (j + 1)) * h_11)

        moments = moments * 256
        print('Intensidad promedio =',moments[0])
        print('Desviación estándar =',moments[1])
        print('Suavidad =',moments[2])
        print('Asimetría =',moments[3])
        print('Uniformidad =',moments[4])
        print('Entropía =',moments[5])

    def recortar2(self):
        img = cv2.cvtColor(self.img,cv2.COLOR_RGB2YUV)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img = cv2.cvtColor(img,cv2.COLOR_YUV2BGR)
    
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #img[:,:,0] = clahe.apply(img[:,:,0])


        # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # # print(img)
        # # Convert to gray-scale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
        # # Blur the image to reduce noise
        hBlur = cv2.medianBlur(h, 11)
        hBlur = cv2.medianBlur(hBlur, 11)
        #hBlur = cv2.medianBlur(hBlur, 11)
        ###graficar(hBlur)

        ret2,_ = cv2.threshold(hBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2,_ = cv2.threshold(hBlur[hBlur>ret2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2,_ = cv2.threshold(hBlur[hBlur>ret2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        print("umbral:",ret2)
        mascara=hBlur.copy()
        mascara[mascara<=ret2]=0
        mascara[mascara!=0]=1

        kernel1 = np.ones((7,7),np.uint8)### agrego kernel de las morfologicas
        kernel2 = np.ones((27,27),np.uint8)
        mascara=cv2.dilate(mascara, kernel1,iterations = 1)
        mascara=cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel2)

        mascara=mascara.astype(np.bool_)## remove_small solo funciona con variables binarias
        mascara=morphology.remove_small_objects(mascara,6000,connectivity=10,in_place=True)

        mascara=mascara.astype(np.uint8)

        # kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(71,71))
        # # mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        # mascara=cv2.dilate(mascara,kernel,iterations = 1)
        #graficar(mascara)
        #print(mascara.shape)
        valox=mascara.shape
        #print('este es x ',valox[1])
        #     mascara=mascara*255
        yhisto=mascara.sum(axis=0)/len(mascara)
        xhisto=mascara.sum(axis=1)/len(mascara)
        # if xhisto.max()<yhisto.max():
        #     yhisto=mascara[:yhisto.max(),:].sum(axis=0)
        # print(np.where(yhisto==np.max(yhisto))[0][0])
        # macara=mascara[:np.where(yhisto==np.max(yhisto))[0][0],]
        if np.max(xhisto)>np.max(yhisto):
            #print(np.where(xhisto==np.max(xhisto))[0][0])
            index1=np.where(xhisto==np.max(xhisto))[0][0]
            #print('el max es x')
            
            x1= index1+800 ##izq
            x2= index1-800 ##derec

            #print('index1,x1,x2', index1,x1,x2)
            

            if x1>4160:  
                x11= x1-4160
                mascara=mascara[index1-800-x11:index1+800-x11,:] 
            
                img=img[index1-800-x11:index1+800-x11,:] 
                ###graficar(img)

                ###graficar(mascara)
                #print('mayor a 3120')
            
            elif x2<0:
                x22=abs(x2)
                mascara=mascara[index1-800+x22:index1+800+x22,:]

                img=img[index1-800+x22:index1+800+x22,:]
                ###graficar(img)

                ###graficar(mascara)
                #print('menor a 0 en x')

            else:
                mascara=mascara[index1-800:index1+800,:]
                ###plt.plot(xhisto)
                ###plt.show()
                ##plt.plot(yhisto)
                ###graficar(mascara)
            
                img=img[index1-800:index1+800,:]
                ###graficar(img)

                #print('en x ',mascara)
                #print('todo bien')
            
            yhisto=mascara.sum(axis=0)/len(mascara)
            ###plt.plot(yhisto)
            index2=np.where(yhisto==np.max(yhisto))[0][0]
            #print('index',index2)

            y1= index2+800 ##izq
            y2= index2-800 ##derec

            #print('y1 y y2', y1,y2)
        
            if y1>3120:  
                y11= y1-3120
                mascara=mascara[:,index2-800-y11:index2+800-y11] 

                img=img[:,index2-800-y11:index2+800-y11] 
                ###graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ###graficar(img)

                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)
                ###graficar(imgmascara)
            
                #print('mayor a 4160')
            
            elif y2<0:
                y22=abs(y2)
                mascara=mascara[:,index2-800+y22:index2+800+y22]

                img=img[:,index2-800+y22:index2+800+y22]
                ####graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ####graficar(img)

                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)
                ###graficar(imgmascara)

                #print('menor a 0 en y')
        
            else: 
                mascara=mascara[:,index2-800:index2+800]

                img=img[:,index2-800:index2+800]
                ###graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ###graficar(img)

                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)
                ###graficar(imgmascara)
                #img=img[index1-800:index1+800,index2-800:index2+800]
                #graficar(img)
                #print('eSte es ', index2)
                #print('todo bien')          
    
        else:

            #print('el max es y')

            index2=np.where(yhisto==np.max(yhisto))[0][0]
            #print('index2',index2)
            y1= index2+800 ##izq
            y2= index2-800 ##derec

            if y1>3120:
                y11= y1-3120
                mascara=mascara[:,index2-800-y11:index2+800-y11] 

                img=img[:,index2-800-y11:index2+800-y11] 
                ###graficar(img)

                ###graficar(mascara)
                #print('mayor a 4160 2')
            
            elif y2<0:
                y22=abs(y2)
                mascara=mascara[:,index2-800+y22:index2+800+y22]
            
                img=img[:,index2-800+y22:index2+800+y22]
                ###graficar(img)
            
                #print('menor a 0 en y 2')
                ###graficar(mascara)

            else: 
                mascara=mascara[:,index2-800:index2+800]
                ###graficar(mascara)
                #print('todo bien')

                img=img[:,index2-800:index2+800]
                ###graficar(img)
            
            xhisto=mascara.sum(axis=1)/len(mascara)
            #plt.plot(yhisto)
            index1=np.where(xhisto==np.max(xhisto))[0][0]
        
        
            x1= index1+800 ##izq
            x2= index1-800 ##derec
        
            if x1>4160:  
                x11= x1-4160
                mascara=mascara[index1-800-x11:index1+800-x11,:] 

                img=img[index1-800-x11:index1+800-x11,:] 
                ###graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ###graficar(img)
        
                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)
                ###graficar(imgmascara)

                #print('mayor a 3120 2')
            
            
            elif x2<0:
                x22=abs(x2)
                mascara=mascara[index1-800+x22:index1+800+x22,:]

                img=img[index1-800+x22:index1+800+x22,:]
                ###graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ####graficar(img)

                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)
                ###graficar(imgmascara)

                #print('menor a 0 en x 2')

            else:
                mascara=mascara[index1-800:index1+800,:]

                img=img[index1-800:index1+800,:]
                ###graficar(img)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ###graficar(img)

                ###graficar(mascara)
                imgmascara= cv2.bitwise_and(img,img,mask = mascara)

                ###graficar(imgmascara)
                #print('todo bien')
                #img=img[index1-800:index1+800,index2-800:index2+800]
                #graficar(img)
        origi = pg.image(self.img)
        self.commonMask = mascara
        self.commonImage = imgmascara
        origi.setWindowTitle("Imagen Original")
        filtered = pg.image(imgmascara)
        filtered.setWindowTitle("Imagen Segmentada")
        filtered = pg.image(mascara)
        filtered.setWindowTitle("Imagen Máscara")
            
    def recortar(self):
        imag = cv2.cvtColor(self.img,cv2.COLOR_RGB2YUV)
        
        imag[:,:,0] = cv2.equalizeHist(imag[:,:,0])
        imag = cv2.cvtColor(imag, cv2.COLOR_YUV2BGR)
        h = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)[:,:,0]
        # # Blur the image to reduce noise
        hBlur = cv2.medianBlur(h, 11)
        ret2,_ = cv2.threshold(hBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret2,_ = cv2.threshold(hBlur[hBlur>ret2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print("umbral:",ret2)
        mascara=hBlur.copy()
        mascara[mascara<=ret2]=0
        mascara[mascara!=0]=1
        mascara=mascara.astype(np.bool_)## remove_small solo funciona con variables binarias
        mascara=morphology.remove_small_objects(mascara,5000,connectivity=10,in_place=True)
        mascara=mascara.astype(np.uint8)
        
        yhisto=mascara.sum(axis=0)/len(mascara)
        xhisto=mascara.sum(axis=1)/len(mascara)
        
        if np.max(xhisto)>np.max(yhisto):
            index1=np.where(xhisto==np.max(xhisto))[0][0]
            mascara=mascara[index1-800:index1+800,:]            
            yhisto=mascara.sum(axis=0)/len(mascara)
            index2=np.where(yhisto==np.max(yhisto))[0][0]
            mascara=mascara[:,index2-800:index2+800]
            imag=imag[index1-800:index1+800,index2-800:index2+800]
        else:
            
            index2=np.where(yhisto==np.max(yhisto))[0][0]
            mascara=mascara[:,index2-800:index2+800]
            xhisto=mascara.sum(axis=1)/len(mascara)
            index1=np.where(xhisto==np.max(xhisto))[0][0]
            mascara=mascara[index1-800:index1+800,:]
            imag=imag[index1-800:index1+800,index2-800:index2+800]
            
        imag = imgmascara = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        imgmascara= cv2.bitwise_and(imag,imag,mask = mascara)
        
        # Print images
        self.imag = imag
        origi = pg.image(self.img)
        origi.setWindowTitle("Imagen Original")
        filtered = pg.image(imgmascara)
        filtered.setWindowTitle("Imagen Segmentada")
        
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
        self.wGlob.plotHisto.plot(x,hist, title=('Histograma Global')) 
        self.wGlob.plotHisto.setLabel('left','Cantidad de pixeles' )
        self.wGlob.plotHisto.setLabel('bottom','Intensidad de iluminacion' )
        

            
             
  
if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    
    
    