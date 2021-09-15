#!/usr/bin/env python
# coding: utf-8

# # 13-7-21 AIPIA_TFM_UNIR_VPAP

# ## MODULO VIDEO  YOLO V2 

# In[ ]:


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2, sys
from yolo2.preprocessing_y2 import parse_annotation, BatchGenerator
from yolo2.utils_y2 import WeightReader, decode_netout, draw_boxes

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


global c, ver_si, persona_reg, placa_reg, caras_cola, ocr_cola, cap,sentido, camara_in, camara_out
c, ver_si=0,0
persona_reg=" "*10
placa_reg=" "*10
caras_cola=0
ocr_cola=0
sentido=0
camara_in=0
camara_out='http://192.168.43.73:8080/video'


# In[ ]:


LABELS = ['persona', '', 'coche', 'moto', '', '', '', 'camion']+['']*72
IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0
BATCH_SIZE       = 16
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50


# In[ ]:


from yolo2.y2 import y2


# In[ ]:


model_y2 = y2()
model_y2.load_weights("yolo2/yolov2.h5")


# In[ ]:


dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
input_image = np.expand_dims(np.zeros((416,416,3)), 0)
skip=model_y2.predict([input_image, dummy_array])


# In[ ]:


from datetime import datetime
def save_boxes(image, boxes, labels):
    global directorio_caras_in, c, caras_cola, directorio_ocr_in, ocr_colas,sentido
    c=0
    now = datetime.datetime.now()
    horas=str(now.hour)+str(now.minute)+str(now.second)+".jpg"
    image_h, image_w, _ = image.shape
    t=0
    if sentido==0:
        dir_caras=directorio_caras_in
        dir_ocr=directorio_ocr_in
    else:
        dir_caras=directorio_caras_out
        dir_ocr=directorio_ocr_out
    for box in boxes:
        numero=box.get_label()
        if numero not in(0,2,3,7): continue
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)
        nombre=labels[numero] +str(t)+horas
        t+=1
        if numero==0:
            nombre=dir_caras + str(nombre)
            caras_cola = len(os.listdir(dir_caras))
            textbox22.delete("1.0","end")
            textbox22.insert('end', caras_cola)
        else:
            nombre=dir_ocr+str(nombre)
            ocr_cola = len(os.listdir(dir_ocr))
            textbox32.delete("1.0","end")
            textbox32.insert('end', ocr_cola)
            
        result=cv2.imwrite(nombre, image[ymin:ymax,xmin:xmax])
    return

def save_boxes2(image, boxes,image_name, labels):
    global directorio_caras_in, c, caras_cola, directorio_ocr_in, ocr_colas,sentido
    c=0
    now = datetime.datetime.now()
    horas=str(now.hour)+str(now.minute)+str(now.second)+".jpg"
    image_h, image_w, _ = image.shape
    t=0
    outfile = open(os.path.join("AP/", image_name[0 : -3] + 'txt'), 'w')
    if sentido==0:
        dir_caras=directorio_caras_in
        dir_ocr=directorio_ocr_in
    else:
        dir_caras=directorio_caras_out
        dir_ocr=directorio_ocr_out
        
    for box in boxes:
        numero=box.get_label()
        if numero not in(0,2,3,7): continue
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)
        nombre=labels[numero] +str(t)+horas
        outfile.write("{} {} {} {} {} {}\n".format(labels[numero], box.get_score(), xmin, ymin, xmax, ymax))
        print("{} {} {} {} {} {}\n".format(labels[numero], box.get_score(), xmin, ymin, xmax, ymax))
        t+=1
        if numero==0:
            nombre=dir_caras + str(nombre)
            caras_cola = len(os.listdir(dir_caras))
            textbox22.delete("1.0","end")
            textbox22.insert('end', caras_cola)
        else:
            nombre=dir_ocr+str(nombre)
            ocr_cola = len(os.listdir(dir_ocr))
            textbox32.delete("1.0","end")
            textbox32.insert('end', ocr_cola)
            
        result=cv2.imwrite(nombre, image[ymin:ymax,xmin:xmax])
    outfile.close()
    return


# ## MODULO DETECTOR MATRICULAS KERAS YOLO v4 ¶
# 

# In[ ]:


import colorsys
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from yolo4.decode_np import Decode


# In[ ]:


from glob import glob
import sqlite3
import time
import datetime
import pytesseract
sys.path.append("..")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


# In[ ]:


def get_class_y4(classes_path_y4):
    classes_path_y2 = os.path.expanduser(classes_path_y4)
    with open(classes_path_y4) as f:
        class_names_y4 = f.readlines()
    class_names_y4 = [c.strip() for c in class_names_y4]
    return class_names_y4

def get_anchors_y4(anchors_path_y4):
    anchors_path_y4 = os.path.expanduser(anchors_path_y4)
    with open(anchors_path_y4) as f:
        anchors_y4 = f.readline()
    anchors_y4 = [float(x) for x in anchors_y4.split(',')]
    return np.array(anchors_y4).reshape(-1, 2)


# In[ ]:


model_path_y4 = 'yolo4/placas.h5'
anchors_path_y4 = 'yolo4/model_data/yolo4_anchors.txt'
classes_path_y4 = 'yolo4/model_data/placa_classes.txt'
class_names_y4 = get_class_y4(classes_path_y4)
anchors_y4 = get_anchors_y4(anchors_path_y4)
num_anchors_y4 = len(anchors_y4)
num_classes_y4 = len(class_names_y4)
model_image_size_y4 = (608, 608)
conf_thresh_y4 = 0.2
nms_thresh_y4 = 0.45


# In[ ]:


yolo4_model = yolo4_body(Input(shape=model_image_size_y4+(3,)), num_anchors_y4//3, num_classes_y4)
model_path_y4 = os.path.expanduser(model_path_y4)
assert model_path_y4.endswith('.h5'), 'Keras modelo o pesos deben tener el formato de archivo .h5'
yolo4_model.load_weights(model_path_y4)


# In[ ]:





# In[ ]:


def ocr(img, data,nombre_placa_box):
    xmin, ymin, xmax, ymax, classes,  scores, ww,hh = data
    #box = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    box = img[int(ymin):int(ymax), int(xmin)+int((xmax-xmin)*0.12):int(xmax)-int((xmax-xmin)*0.015)]
    # metricas de ocr, se puede desactivar
    result_box=cv2.imwrite(nombre_placa_box, box)
    
    
    # metricas de OCR fin
    im = Image.fromarray(box)
    escala=int((200*(ymax-ymin))//(xmax-xmin))
    img_tmp = ImageTk.PhotoImage(image=im.resize((200, escala)))
    panelC.configure(image=img_tmp)
    panelC.image =img_tmp
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # threshold the image using Otsus method to preprocess for tesseract
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform a median blur to smooth image slightly
    blur = cv2.medianBlur(thresh, 3)#3
    # resize image to double the original size as tesseract does better with certain text size
    blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # run tesseract and convert image text to string
    text = pytesseract.image_to_string(blur, config='--psm 12')
    print("Class: {}, Text Extracted: {}".format("Matricula: ", text))
    return text


# In[ ]:


def get_matricula(sentido):
    global base_de_datos
    plate=[]
    if sentido==0: path1= directorio_ocr_in
    else: path1= directorio_ocr_out
    test_batch = os.listdir(path1)[0:1]
    if len(test_batch)==0: return
    matriculas=[]
    for fila_test in test_batch:
        nombre_fichero=path1+fila_test
        nombre_placa_box= "PLACAS/"+ fila_test  # para salvar placas en directorio placas (metricas)
        image_name= 'AP4/' + fila_test[0 : -3] + 'txt'
        try:
            image1 = cv2.imread(nombre_fichero)
            image_aux, boxes, scores, classes1 = _decode.detect_image(image1, True)
            #

            outfile = open(image_name, 'w')
            outfile.write("{} {} {} {} {} {}\n".format("placa", scores[0], 
                        int(boxes[0][0]),int(boxes[0][1]),int(boxes[0][2]),int(boxes[0][3])))

            outfile.close()
            
            #
            if boxes is not None and len(boxes)>0:
                bb=[boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3],classes1[0],scores[0],0,0]
                resulta=ocr(image1,bb,nombre_placa_box)
                matriculas.append(resulta)
            else :
                resulta=''
            img33 = ImageTk.PhotoImage((Image.open(nombre_fichero)).resize((200, 200)))
            panelCC.configure(image=img33)
            panelCC.image =img33
            print("Fichero y matrículas: ",fila_test, resulta,end='\n**************\n')
            os.remove(path1+fila_test)

            placas_cola = len(os.listdir(path1))
            textbox32.delete("1.0","end")
            textbox32.insert('end', placas_cola)
            textbox33.delete("1.0","end")
            textbox34.delete("1.0","end")
        except:
             os.remove(nombre_fichero)
    
    for rows in matriculas:
        plate=[]
        for t in range(len(rows)):
            if rows[t].isdigit() or rows[t].isupper():
                plate.append(rows[t])
        plate= ["".join(plate)]
    if plate is None: 
        img3 = ImageTk.PhotoImage(Image.open("iconos/ud.jpg"))
        panelC.configure(image=img3)
        panelC.image =img3
    
    return plate
    
    
        
def check_matricula(plate, sentido,base_de_datos):
    
    con = sqlite3.connect(base_de_datos)
    valor=sql_matricula(con,plate)
    if valor>0:
        
        #abre=directorio_caras_reg + str(plate[0])+".jpg"
        #print("Empleado identificado y autorizado",abre)
        textbox34.insert('end', "Matrícula identificada, vehículo autorizado\n")
        #img20 = ImageTk.PhotoImage((Image.open(abre)).resize((200, 200)))
        #panelB.configure(image=img20)
        #panelB.image =img20
                
        print("Matricula Registrada en la base de Datos: ",valor)
        aparcado_si=sql_parking(con,plate)
        if aparcado_si==0:
            aparcado_plate_in(con,plate) # vehículo que aun No ha aparcado, entra
            textbox34.insert('end', "Vehículo  sin aparcar, acceso abierto\n")
            print("NO ha aparcado aun, a aparcar")
        else:
            print("El vehículo ya está aparcado")
            textbox34.insert('end', "Vehículo  dentro del parking\n")
            if sentido==1:
                aparcado_plate_out(con,plate) # vehículo aparcado, sale
                print("Vehículo saliendo")
                textbox34.insert('end', "Vehículo saliendo del parking, salida permitida\n")
    else:
        print("Matricula No registrada, acceso NO permitido")
        textbox34.insert('end', "Matricula No registrada, acceso NO permitido\n")
    con.close()
        
    return plate


# In[ ]:


def sql_matricula(con,placa):
    cursorObj = con.cursor()
    args=placa 
    sql="SELECT * FROM VEHICULOS WHERE matricula IN ({seq})".format(
    seq=','.join(['?']*len(args)))
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    print(rows1)
    return(len(rows1))


# In[ ]:


def parking_update(con):

    cursorObj = con.cursor()
    cursorObj.execute('UPDATE PARKING SET HORAS= HORA_OUT - HORA_IN  ')
    con.commit()


# In[ ]:


def sql_parking(con,placa):   # Miramos si ya ha fichado la entrada hoy
    cursorObj = con.cursor()
    args=[placa[0]]
    args.append(str(datetime.date.today()))
    sql="SELECT * FROM PARKING WHERE MATRICULA IN ({}) AND FECHA == ({}) ".format(
    ','.join(['?']),
    ','.join(['?']))
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    print("MIrando si ha aparcado: ",rows1,args)
    return(len(rows1))


# In[ ]:


def aparcado_plate_in(con,placa):

    A = time.ctime()
    A=A.split()
    hora=int(A[3][0:2])
    minuto=int(A[3][3:5])
    print (hora, minuto)
    cursorObj = con.cursor()
    data = [( datetime.date.today(), placa[0],hora,22,0)]
    cursorObj.executemany("INSERT INTO PARKING VALUES(?, ?, ?,?,?)", data)
    con.commit()
    parking_update(con)

def aparcado_plate_out(con,placa):

    A = time.ctime()
    A=A.split()
    hora=int(A[3][0:2])
    minuto=int(A[3][3:5])
    print (hora, minuto)
    cursorObj = con.cursor()
    args=[hora]
    args.append(placa[0])
    sql="UPDATE PARKING SET HORA_OUT = {} where MATRICULA == {}".format(
    ','.join(['?']),
    ','.join(['?']))
    cursorObj.execute(sql,args)
    con.commit()
    parking_update(con)


# In[ ]:


def ocr_on():
    global placa_reg, base_de_datos, sentido
    plate=get_matricula(sentido)# sentido 'IN', 'OUT
    if plate:
        check_matricula(plate,sentido, base_de_datos)# sentido 'IN', 'OUT
        textbox33.insert('end', plate)
        print(plate)

def ocr_off():
    pass

def ver_ocr_on():
    pass


def ver_ocr_off():
    pass


# In[ ]:


_decode = Decode(conf_thresh_y4, nms_thresh_y4, model_image_size_y4, yolo4_model, class_names_y4)


# In[ ]:





# # MODULO GUI MARCOS + GRID + POPUP + MENU

# In[ ]:


import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import scrolledtext as st
from tkinter import Menu
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

import imutils
import pandas as pd
import sqlite3

import shutil
import win32api
import win32print
import tempfile


# In[ ]:


global base_de_datos
global directorio_actual
global directorio_caras_reg
global directorio_caras_in
global directorio_caras_out
global directorio_ocr_in
global directorio_ocr_out
global directorio_fotos_in
global directorio_fotos_out

directorio_actual= os.getcwd()
base_de_datos=directorio_actual + "\\tfm.db"
directorio_caras_reg=directorio_actual + "\\C_REG\\"
directorio_caras_in=directorio_actual + "\\C_IN\\"
directorio_caras_out=directorio_actual + "\\C_OUT\\"
directorio_ocr_in=directorio_actual + "\\V_IN\\"
directorio_ocr_out=directorio_actual + "\\V_OUT\\"
directorio_fotos_in=directorio_actual + "\\F_IN\\"
directorio_fotos_out=directorio_actual + "\\F_OUT\\"

print(directorio_actual, base_de_datos,directorio_caras_reg,directorio_caras_in,directorio_caras_out,directorio_ocr_in,directorio_ocr_out)
global printerdef
printerdef = ""


# In[ ]:


def abrir():
    global base_de_datos
    base_de_datos = filedialog.askopenfilename(initialdir = directorio_actual,
                title = "Seleccione base de datos",filetypes = (("db","*.db"),
                ("Cualquiera","*.*")),defaultextension= ".bd")
    cuadro_texto_base_actualiza()

def crear():
    global base_de_datos
    base_de_datos = filedialog.asksaveasfilename(initialdir = directorio_actual,
            title = "Guardar como",filetypes = (("db","*.db"), 
            ("Cualquiera","*.*")),defaultextension= ".bd")
    shutil.copyfile('TFM_pl.db', base_de_datos)
    cuadro_texto_base_actualiza()

def grabar():
    global base_de_datos
    base_de_datos2 = filedialog.asksaveasfilename(initialdir = directorio_actual,
            defaultextension= ".bd",title = "Guardar como",filetypes = (("db","*.db"),
            ("Cualquiera","*.*")))
    shutil.copyfile(base_de_datos, base_de_datos2)
    base_de_datos=base_de_datos2
    cuadro_texto_base_actualiza()

def carpeta_base():
    global directorio_actual
    directorio_actual=cambia_directorio(directorio_actual)
    
def cambia_directorio(dir_tmp_in):

    dir_tmp_out=filedialog.askdirectory(initialdir = dir_tmp_in)
    if dir_tmp_out!="":
        os.chdir(dir_tmp_out)
    print(os.getcwd())
    return dir_tmp_out

def salir():
    window.destroy()
    
    
def cam_in():
    pt = Toplevel()
    pt.geometry("200x100")
    pt.title("Camara de Entrada")
    var1 = StringVar()
    LABEL = Label(pt, text="Selecciona Dispositivo").pack()
    PRCOMBO = ttk.Combobox(pt, width=35,textvariable=var1)

    PRCOMBO["values"] = [0,'http://192.168.43.73:8080/video','http://admin:holahola@192.168.43.28/video.cgi?.mjpg']
    PRCOMBO.pack()
    def select():
        global camara_in
        camara_in = PRCOMBO.get()
        if camara_in=='0': camara_in=0
        print(camara_in)
        pt.destroy()
    BUTTON = ttk.Button(pt, text="OK",command=select).pack()

def cam_out():
    pt = Toplevel()
    pt.geometry("200x100")
    pt.title("Camara de Salida")
    var1 = StringVar()
    LABEL = Label(pt, text="Selecciona Dispositivo").pack()
    PRCOMBO = ttk.Combobox(pt, width=35,textvariable=var1)

    PRCOMBO["values"] = [0,'http://192.168.43.73:8080/video','http://admin:holahola@192.168.43.28/video.cgi?.mjpg']
    PRCOMBO.pack()
    def select():
        global camara_out
        camara_in = PRCOMBO.get()
        if camara_in=='0': camara_out=0
        print(camara_out)
        pt.destroy()
    BUTTON = ttk.Button(pt, text="OK",command=select).pack()

def dir_rostros_reg():
    global directorio_caras_reg 
    directorio_caras_reg=cambia_directorio(directorio_caras_reg)
    print(directorio_caras_reg)
    
def dir_rostros_in():
    global directorio_caras_in 
    directorio_caras_in=cambia_directorio(directorio_caras_in)
    print(directorio_caras_in)

def dir_rostros_out():
    global directorio_caras_out 
    directorio_caras_out=cambia_directorio(directorio_caras_out)
    print(directorio_caras_out)
    
def sobre():
    mb.showinfo('Acerca de', 'AIPIA v1\n 2021 copyright')

def impresora_local():
    pt = Toplevel()
    pt.geometry("200x100")
    pt.title("Menu Impresión")
    var1 = StringVar()
    LABEL = Label(pt, text="Selecciona Impresora").pack()
    PRCOMBO = ttk.Combobox(pt, width=35,textvariable=var1)
    print_list = []
    printers = list(win32print.EnumPrinters(2))
    for i in printers:
        print_list.append(i[2])
    PRCOMBO["values"] = print_list
    PRCOMBO.pack()
    def select():
        global printerdef
        printerdef = PRCOMBO.get()
        pt.destroy()
    BUTTON = ttk.Button(pt, text="OK",command=select).pack() 


# In[ ]:


def sql_fichajes():
    global base_de_datos
    con = sqlite3.connect(base_de_datos)
    df = pd.read_sql_query("SELECT * from FICHAJE", con)
    textbox10.insert('end', '\n ****** FICHAJE ***** \n')
    textbox10.insert('end', df)
    textbox10.insert('end', '\n')
    con.close()
    
def sql_registro():
    global base_de_datos
    con = sqlite3.connect(base_de_datos)
    df = pd.read_sql_query("SELECT * from registro", con)
    textbox10.insert('end', '\n ****** REGISTRO ***** \n')
    textbox10.insert('end', df)
    textbox10.insert('end', '\n')
    con.close()
    
def limpiacuadrotexto():
    textbox10.delete("1.0","end")
    
def cuadro_texto_base_actualiza():
    global base_de_datos
    textbox30.delete("1.0","end")
    textbox30.insert('end', base_de_datos)
    
def imprimecuadro():
    global printerdef
    printText = textbox10.get("1.0", END)
    filenameprint = tempfile.mktemp(".txt")
    open(filenameprint, "w").write(printText)
    win32api.ShellExecute(0,"printto", filenameprint,
    "%s" % win32print.GetDefaultPrinter(),".",0 )
    


# In[ ]:





# In[ ]:


def visualizar_fotos():
    global ver_si
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    path3= directorio_fotos_in
    test_batch = os.listdir(path3)[0:1]
    if len(test_batch)==0:   return
    for fila_test in test_batch:
        nombre_fichero=path3+fila_test      

        frame1 = cv2.imread(nombre_fichero)
        input_image = cv2.resize(frame1, (416, 416))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = model_y2.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0], 
                              obj_threshold=0.3,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS, 
                              nb_class=CLASS)

        save_boxes2(frame1, boxes,fila_test, labels=LABELS)
        if ver_si:
            frame1 = draw_boxes(frame1, boxes, labels=LABELS)
            frame1 = imutils.resize(frame1, width=480)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame1)
            img = ImageTk.PhotoImage(image=im)
            panelA.configure(image=img)
            panelA.image = img
        else:
            panelA.image = ImageTk.PhotoImage(Image.open("iconos/aipia.jpg"))
        print("REmueve fichero",nombre_fichero)
        os.remove(nombre_fichero)


# In[ ]:



def visualizar():
    global cap
    global c, ver_si
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    frameRate = 10
    
    if cap is not None:
        ret, frame1 = cap.read()
        if ret == True:
            input_image = cv2.resize(frame1, (416, 416))
            input_image = input_image / 255.
            input_image = input_image[:,:,::-1]
            input_image = np.expand_dims(input_image, 0)
            netout = model_y2.predict([input_image, dummy_array])
            boxes = decode_netout(netout[0], 
                                  obj_threshold=0.3,
                                  nms_threshold=NMS_THRESHOLD,
                                  anchors=ANCHORS, 
                                  nb_class=CLASS)
            
            if(c % frameRate == 0):
                save_boxes(frame1, boxes, labels=LABELS)
            c+=1
            if ver_si:
                frame1 = draw_boxes(frame1, boxes, labels=LABELS)
                frame1 = imutils.resize(frame1, width=480)
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame1)
                img = ImageTk.PhotoImage(image=im)
                panelA.configure(image=img)
                panelA.image = img
            panelA.after(10, visualizar)
            
        else:
            panelA.image = ImageTk.PhotoImage(Image.open("iconos/aipia.jpg"))
            cap.release()


# In[ ]:


def yolo2_fotos_on():
    global sentido, camara_in,camara_out, fotos_on_off
    
    if fotos_on_off==0:
        #button21.config(image=photo21)
        #fotos_on_off=1
        visualizar_fotos()
    else:
        #button21.config(image=photo20)
        #fotos_on_off=0
        finalizar1()


def yolo2_on():
    global cap,sentido, camara_in,camara_out, video_on_off
    
    if video_on_off==0:
        button20.config(image=photo21)
        video_on_off=1
        if sentido==0: cap = cv2.VideoCapture(camara_in)
        else: cap = cv2.VideoCapture(camara_out)
        visualizar()
    else:
        button20.config(image=photo20)
        video_on_off=0
        yolo2_off()


def finalizar1():

    img2 = ImageTk.PhotoImage(Image.open("iconos/aipia.jpg"))
    panelA.configure(image=img2)
    panelA.image =img2

def video_on():
    global ver_si
    
    if ver_si==0:
        button22.config(image=photo21)
        ver_si=1
    else:
        button22.config(image=photo20)
        ver_si=0



def yolo2_off():
    global cap
    cap.release()
    panelA.after(50,finalizar1)

def video_off():
    global ver_si
    ver_si=0


# In[ ]:


def caras_on():
    global persona_reg, sentido
    denei1=get_dni(sentido)
    if denei1:
        check_dni(denei1, sentido)
        textbox23.insert('end', denei1)

def caras_off():
    pass

def ver_caras_on():
    pass


def ver_caras_off():
    pass


# # MODULO DETECTOR ROSTROS  DNI

# In[ ]:



import face_recognition

import sqlite3
import time
import datetime


# In[ ]:


path1=directorio_caras_reg
images_reg = os.listdir(path1)[:]


# In[ ]:


encodings_conocidos=[]
nombres_conocidos=[]
for file in images_reg:
     #Cargamos la base de datos con las caras identificar:
    imagen_flujo = face_recognition.load_image_file(path1+file)
    imagen_encodings = face_recognition.face_encodings(imagen_flujo)[0]  
    encodings_conocidos.append(imagen_encodings)
    nombres_conocidos.append(file)


# In[ ]:


def detecta_caras(img_ocr): 
    loc_rostros = [] #Localizacion de los rostros en la imagen (contendrá las coordenadas de los recuadros que las contienen)
    encodings_rostros = [] #Encodings de los rostros
    nombres_rostros = [] #Nombre de la persona de cada rostro 
    loc_rostros = face_recognition.face_locations(img_ocr) #Localizamos cada rostro
    encodings_rostros = face_recognition.face_encodings(img_ocr, loc_rostros) #extraemos sus encodings:
    for encoding in encodings_rostros: #Recorre el array de encodings encontrado:
        coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding) #Coincide con encoding conocido?
        if True in coincidencias:  # coincidencias es un array booleano (True o False)
            nombre = nombres_conocidos[coincidencias.index(True)] #Busca dni en el array conocidos
        else:
            nombre = "?"
        nombres_rostros.append(nombre)  #Añade DNI identificado al array de nombres
    #dibuja_box(img,encodings_conocidos,encodings_rostros,loc_rostros,nombres_rostros)
    
    if len(nombres_rostros)>0: return [(nombres_rostros[0])[0:-4]]


# In[ ]:


def dibuja_box_ocr(img,encodings_conocidos,encodings_rostros,loc_rostros,nombres_rostros):
    font = cv2.FONT_HERSHEY_COMPLEX #Cargamos una fuente de texto:
    #Dibuja recuadro rojo alrededor de los rostros desconocidos, y uno verde alrededor de los conocidos:
    for (top, right, bottom, left), nombre in zip(loc_rostros, nombres_rostros):

        if nombre != "?":
            color = (0,255,0) #Verde
        else:
            color = (255,0,0) #Rojo

        #Dibujar los recuadros alrededor del rostro:
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)

        #Escribir el nombre de la persona:
        cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0,0,0), 1)

    #Abrimos una ventana con el resultado:
    cv2.imshow('Output',  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:


def tabla_update(con,tabla):
    cursorObj = con.cursor()
    sql='UPDATE '+ tabla + ' SET HORAS= HORA_OUT - HORA_IN  '
    cursorObj.execute(sql)
    sql='UPDATE '+ tabla + ' SET EXTRAS= HORAS-8 WHERE HORAS>=8 '
    cursorObj.execute(sql)
    con.commit()


# In[ ]:


def fichaje_dni_in(con,denei,tabla):

    dia = datetime.datetime.today()
    cursorObj = con.cursor()
    data = [( datetime.date.today(), denei[0],dia.hour,dia.minute,-1,0,0,0)]
    cursorObj.executemany("INSERT INTO "+tabla+" VALUES(?, ?, ?,?,?,?,?,?)", data)
    con.commit()
    tabla_update(con,tabla) #tabla_update(con,"FICHAJE")
    return dia

def sql_tfno(con,denei,tabla="empleados"):

    cursorObj = con.cursor()
    args=[str(denei[0])]
    print("Argumentos malos:",denei[0])
    sql='SELECT TELEFONO FROM '+ tabla + ' WHERE DNI IN ({})'.format(
    ','.join(['?']))
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    print("El telefono es :",rows1)
    aaa="el telefono es :" + str(rows1) +"\n"
    textbox24.insert('end',aaa)
    #numbers = [int(temp)for temp in rows1[0][0].split() if temp.isdigit()]
    
    print (rows1[0][0])
    return rows1[0][0]


# In[ ]:


def fichaje_dni_out(con,denei,tabla):
    dia = datetime.datetime.today()
    cursorObj = con.cursor()
    args = [dia.hour,dia.minute,denei[0],str(datetime.date.today())]
    sql="UPDATE " + tabla + " SET HORA_OUT = {} , MIN_OUT ={}  where dni == {} AND FECHA == {} AND HORA_OUT==-1 ".format(
    ','.join(['?']),
    ','.join(['?']),
    ','.join(['?']),
    ','.join(['?']))
           
    cursorObj.execute(sql,args)
    rows = cursorObj.rowcount
    if rows>0:
        print("Y ahora se pira, adios", rows)
    else:
        print("Ya se había tramitado su salida", rows)
    con.commit()
    tabla_update(con,tabla)


# In[ ]:


def sql_dni(con,denei,tabla):   # Miramos si el DNI es de un empleado
    cursorObj = con.cursor()
    args=denei
    sql='SELECT * FROM '+tabla + ' WHERE DNI IN ({})'.format(
    ','.join(['?']))
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    print("el DNI es de :",rows1)
    aaa="el DNI es de :" + str(rows1) +"\n"
    textbox24.insert('end',aaa)
    return(len(rows1))


def sql_fichaje(con,denei,tabla):   # Miramos si ha fichado
    cursorObj = con.cursor()
    print("sql_fichaje",denei)
    args=denei
    args.append(str(datetime.date.today()))
    sql="SELECT * FROM "+tabla+" WHERE DNI IN ({}) AND FECHA == ({}) ".format(
    ','.join(['?']),
    ','.join(['?']))   
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    return(len(rows1))

def sql_fichaje_out(con,denei,tabla):
    cursorObj = con.cursor()
    args=[denei]
    args.append(str(datetime.date.today()))
    sql="SELECT * FROM "+tabla+" WHERE DNI IN ({}) AND FECHA == ({}) AND HORA_OUT=-1 ".format(
    ','.join(['?']),
    ','.join(['?']))   
    cursorObj.execute(sql, args)
    rows1 = cursorObj.fetchall()
    return(len(rows1))


# In[ ]:


def get_dni(sentido):
    global caras_cola
    denei=[]
    if sentido==0: path2=directorio_caras_in
    else: path2= directorio_caras_out
    test_batch = os.listdir(path2)[0:1]
    if len(test_batch)==0:   return
    for fila_test in test_batch:
        try:
            img1 = face_recognition.load_image_file(path2+fila_test) #Cargamos la imagen donde hay que identificar los rostros:
            denei=detecta_caras(img1)

            img22 = ImageTk.PhotoImage((Image.open(path2+fila_test)).resize((200, 200)))
            panelBB.configure(image=img22)
            panelBB.image =img22

            print("Fichero y dni: ",fila_test, denei,end='\n**************\n')
            os.remove(path2+fila_test)
            caras_cola = len(os.listdir(path2))
            textbox22.delete("1.0","end")
            textbox22.insert('end', caras_cola)
            textbox23.delete("1.0","end")
            textbox24.delete("1.0","end")
            if denei is None: 
                img2 = ImageTk.PhotoImage(Image.open("iconos/ud.jpg"))
                panelB.configure(image=img2)
                panelB.image =img2
        except:
            os.remove(path2+fila_test)
        
    return denei


# In[ ]:


def check_dni(denei, sentido):   # Miramos si el DNI es de un empleado
    global base_de_datos, cola_whastapp
    con = sqlite3.connect(base_de_datos)  #denei=["09320451T", "09339995V","00000000A"]
    empleado_si=sql_dni(con,denei,"empleados")
    if empleado_si !=0 :
        abre=directorio_caras_reg + str(denei[0])+".jpg"
        print("Empleado identificado y autorizado",abre)
        textbox24.insert('end', "Empleado identificado y autorizado\n")
        img20 = ImageTk.PhotoImage((Image.open(abre)).resize((200, 200)))
        panelB.configure(image=img20)
        panelB.image =img20

        fichado_si=sql_fichaje(con,denei,"FICHAJE")
        if fichado_si==0:
            dia=fichaje_dni_in(con,denei,"FICHAJE") # es de un empleado, que aun No ha fichado, a fichar pues
            mensa="fichaje de entrada del empleado: "+ str(denei)+str(dia)
            telefono=sql_tfno(con, denei, "empleados")
            cola_whastapp.push([telefono,mensa])
            textbox42.delete("1.0","end")
            textbox42.insert('end', cola_whastapp.tamano())
            textbox24.insert('end', "NO ha fichado aun, a fichar\n")
            print("NO ha fichado aun, a fichar")
            print("Cola tamaño: ",cola_whastapp.tamano())
        else:
            print("El empleado ya había fichado")
            textbox24.insert('end', "El empleado ya había fichado\n")
            if sentido==1:
                fichado_out_si=sql_fichaje_out(con,denei[0],"FICHAJE")
                if fichado_out_si==1:
                    dia=fichaje_dni_out(con,denei,"FICHAJE") # empleado que ya ha fichado y se va, adios
                    textbox24.insert('end', "El empleado se marcha, adios\n")
                    mensa="fichaje de Salida del empleado: "+ str(denei)+str(dia)
                    telefono=sql_tfno(con, denei, "empleados")
                    cola_whastapp.push([telefono,mensa])
                    textbox42.delete("1.0","end")
                    textbox42.insert('end', cola_whastapp.tamano())
                else:
                    textbox24.insert('end', "El empleado ya se había marchado\n")
                
    else:
            visita_si=sql_dni(con,denei, "visitas")
            if visita_si !=0 :
                abre=directorio_caras_reg + str(denei[0])+".jpg"
                textbox24.insert('end', "Visitante identificado y autorizado\n")
                img20 = ImageTk.PhotoImage((Image.open(abre)).resize((200, 200)))
                panelB.configure(image=img20)
                panelB.image =img20
                registrado_si=sql_fichaje(con,denei,"REGISTRO")
                if registrado_si==0:
                    dia=fichaje_dni_in(con,denei,"REGISTRO") # es de una visita, que aun No se ha registrado,a regitrar pues
                    mensa="Registro de entrada del visitante: "+ str(denei)+str(dia)
                    cola_whastapp.push([denei[0],mensa])
                    textbox42.delete("1.0","end")
                    textbox42.insert('end', cola_whastapp.tamano())
                    textbox24.insert('end', "NO ha registrado aun, a registrar\n")
                    print("NO ha registrado aun, a registrar")
                                        
                else:
                    #dia=fichaje_dni_in(con,denei,"REGISTRO") # es de una visita, que aun No se ha registrado,a regitrar pues
                    print("La visita ya estaba registrada")
                    textbox24.insert('end', "El visitante ya ha sido registrado\n")
                    if sentido==1:                          
                        registrado_out_si=sql_fichaje_out(con,denei[0],"REGISTRO")
                        if registrado_out_si==1:

                            dia=fichaje_dni_out(con,denei,"REGISTRO") # hay que cambiar esto
                            textbox24.insert('end', "El visitante se marcha, adios\n")
                            mensa="Registro de entrada del visitante: "+ str(denei)+str(dia)
                            cola_whastapp.push([denei[0],mensa])
                            textbox42.delete("1.0","end")
                            textbox42.insert('end', cola_whastapp.tamano())
                        else:
                            textbox24.insert('end', "El visitante ya se había marchado\n")
            else:
                print("Persona NO autorizada")
                textbox24.insert('end', "Persona NO autorizada, prohibida la entrada")
                img20 = ImageTk.PhotoImage((Image.open("iconos/ud.jpg")).resize((200, 200)))
                panelB.configure(image=img20)
                panelB.image =img20
    con.close()


# In[ ]:





# # MODULO COMUNICACION WHASTAPP

# In[ ]:


import pywhatkit as kit
import time
import pyautogui as pg
import subprocess


# In[ ]:


class Cola:
    def __init__(self):
        self.items = []

    def estaVacia(self):
        return self.items == []

    def push(self, item):
        self.items.insert(0,item)

    def pop(self):
        return self.items.pop()

    def tamano(self):
        return len(self.items)


# In[ ]:


def envia_whastapp(a):

    telefono="+34"+ str(a[0])
    #kit.sendwhatmsg(telefono, a[1])
    kit.sendwhatmsg_instantly(telefono, a[1], wait_time=5)
    time.sleep(1)
    pg.hotkey('ctrl', 'w')
    textbox44.insert('end', "Enviado whastapp al movil:"+telefono+" con le mensaje: "+a[1])
    textbox44.delete("1.0","end")


# In[ ]:


def whastapp_on():
    global cola_whastapp
    while cola_whastapp.tamano() >0:
        a=cola_whastapp.pop()
        print(a)
        envia_whastapp(a)
        textbox42.delete("1.0","end")
        textbox42.insert('end', cola_whastapp.tamano())
        
        #subprocess.Popen(["python", "app.py"])


# In[ ]:





# In[ ]:





# #  VENTANA EMERGENTE BASE DE DATOS

# In[ ]:


import empleados_v1


# In[ ]:


class MyDialog:
    def __init__(self, parent, valor, title, labeltext = '' ):
        self.valor = valor
 
        self.top = Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        if len(title) > 0: self.top.title(title)
        if len(labeltext) == 0: labeltext = 'VENTANA NUEVA'
        Label(self.top, text=labeltext).pack()
        self.top.bind("<Return>", self.ok)
        self.e = Entry(self.top, text=valor.get())
        self.e.bind("<Return>", self.ok)
        self.e.bind("<Escape>", self.cancel)
        self.e.pack(padx=15)
        self.e.focus_set()
        b = Button(self.top, text="OK", command=self.ok)
        b.pack(pady=5)


# In[ ]:


class FormularioEmpleados:

    def __init__(self):
        self.empleado1=empleados_v1.Empleados()
        self.ventana1=tk.Tk()
        self.ventana1.title("Mantenimiento de base AIPIA")
        self.cuaderno1 = ttk.Notebook(self.ventana1)        
        self.carga_empleados()
        self.consulta_por_codigo()
        self.listado_completo()
        self.cuaderno1.grid(column=0, row=0, padx=10, pady=10)
        buttonExample = tk.Button(self.ventana1, text="Create new window", command=self.createNewWindow)
        buttonExample.grid(column=0, row=1, padx=4, pady=4)
        but1=tk.Button(self.ventana1, text="cambiar valor", command=self.dialogo)
        but1.grid(column=0, row=2, padx=4, pady=4)      
        
    def createNewWindow(self):
        newWindow = tk.Toplevel(self.ventana1)
        
    def dialogo(self):
        self.valor = StringVar()
        self.valor.set("Hola Manejando datos")
        d = MyDialog(self.ventana1, self.valor, "Probando Dialogo", "VENTANA HIJA")
        root.wait_window(d.top)
                
    def carga_empleados(self):
        self.pagina1 = ttk.Frame(self.cuaderno1)
        self.cuaderno1.add(self.pagina1, text="Carga de Empleados")
        self.labelframe1=ttk.LabelFrame(self.pagina1, text="Empleado")        
        self.labelframe1.grid(column=0, row=0, padx=5, pady=10)
        self.label0=ttk.Label(self.labelframe1, text="DNI")
        self.label0.grid(column=0, row=0, padx=4, pady=4)
        self.dni=tk.StringVar(self.labelframe1)
        self.entrydni=ttk.Entry(self.labelframe1, textvariable=self.dni)
        self.entrydni.grid(column=1, row=0, padx=4, pady=4)
        self.label1=ttk.Label(self.labelframe1, text="Nombre:")
        self.label1.grid(column=0, row=1, padx=4, pady=4)
        self.nombre=tk.StringVar(self.labelframe1)
        self.entrynombre=ttk.Entry(self.labelframe1, textvariable=self.nombre)
        self.entrynombre.grid(column=1, row=1, padx=4, pady=4)
        self.label2=ttk.Label(self.labelframe1, text="Apellidos:")        
        self.label2.grid(column=0, row=2, padx=4, pady=4)
        self.apellidos=tk.StringVar(self.labelframe1)
        self.entryapellidos=ttk.Entry(self.labelframe1, textvariable=self.apellidos)
        self.entryapellidos.grid(column=1, row=2, padx=4, pady=4)
        self.label3=ttk.Label(self.labelframe1, text="CIF:")        
        self.label3.grid(column=0, row=3, padx=4, pady=4)
        self.cif=tk.StringVar(self.labelframe1)
        self.entrycif=ttk.Entry(self.labelframe1, textvariable=self.cif)
        self.entrycif.grid(column=1, row=3, padx=4, pady=4)
        self.label4=ttk.Label(self.labelframe1, text="Móvil:")        
        self.label4.grid(column=0, row=4, padx=4, pady=4)
        self.movil=tk.StringVar(self.labelframe1)
        self.entrymovil=ttk.Entry(self.labelframe1, textvariable=self.movil)
        self.entrymovil.grid(column=1, row=4, padx=4, pady=4)
        self.label5=ttk.Label(self.labelframe1, text="mail:")        
        self.label5.grid(column=0, row=5, padx=4, pady=4)
        self.mail=tk.StringVar(self.labelframe1)
        self.entrymail=ttk.Entry(self.labelframe1, textvariable=self.mail)
        self.entrymail.grid(column=1, row=5, padx=4, pady=4)
        self.label6=ttk.Label(self.labelframe1, text="Departamento:")        
        self.label6.grid(column=0, row=6, padx=4, pady=4)
        self.departamento=tk.StringVar(self.labelframe1)
        self.entrydepartamento=ttk.Entry(self.labelframe1, textvariable=self.departamento)
        self.entrydepartamento.grid(column=1, row=6, padx=4, pady=4)
        self.label7=ttk.Label(self.labelframe1, text="Cargo:")        
        self.label7.grid(column=0, row=7, padx=4, pady=4)
        self.cargo=tk.StringVar(self.labelframe1)
        self.entrycargo=ttk.Entry(self.labelframe1, textvariable=self.cargo)
        self.entrycargo.grid(column=1, row=7, padx=4, pady=4)
        
        self.boton1=ttk.Button(self.labelframe1, text="Trabajador", command=self.agregar1)
        self.boton1.grid(column=1, row=8, padx=4, pady=4)
        self.boton2=ttk.Button(self.labelframe1, text="Visitante", command=self.agregar2)
        self.boton2.grid(column=0, row=8, padx=4, pady=4)

    def agregar1(self):
        datos=(self.dni.get(),self.nombre.get(), self.apellidos.get(), self.cif.get(), self.movil.get(), 
               self.mail.get(), self.departamento.get(), self.cargo.get())
        self.empleado1.alta(base_de_datos,datos,"T")
        mb.showinfo("Información", "Empleado Dado de Alta")
        self.dni.set("")
        self.nombre.set("")
        self.apellidos.set("")
        self.cif.set("")
        self.movil.set("")
        self.mail.set("")
        self.departamento.set("")
        self.cargo.set("")

    def agregar2(self):
        datos=(self.dni.get(),self.nombre.get(), self.apellidos.get(), self.cif.get(), self.movil.get(), 
               self.mail.get(), self.departamento.get(), self.cargo.get())
        self.empleado1.alta(base_de_datos,datos,"V")
        mb.showinfo("Información", "Visitante Dado de Alta")
        self.dni.set("")
        self.nombre.set("")
        self.apellidos.set("")
        self.cif.set("")
        self.movil.set("")
        self.mail.set("")
        self.departamento.set("")
        self.cargo.set("")             
        
    def consulta_por_codigo(self):

        self.pagina2 = ttk.Frame(self.cuaderno1)
        self.cuaderno1.add(self.pagina2, text="Consulta por DNI")
        
        self.labelframe2=ttk.LabelFrame(self.pagina2, text="Empleado")
        self.labelframe2.grid(column=0, row=0, padx=5, pady=10)
        
        self.label02=ttk.Label(self.labelframe2, text="DNI")
        self.label02.grid(column=0, row=0, padx=4, pady=4)
        self.dni2= StringVar(self.labelframe2)
        self.entrydni2=ttk.Entry(self.labelframe2, textvariable=self.dni2)
        self.entrydni2.grid(column=1, row=0, padx=4, pady=4)
        self.label12=ttk.Label(self.labelframe2, text="Nombre:")
        self.label12.grid(column=0, row=1, padx=4, pady=4)
        self.nombre2=tk.StringVar(self.labelframe2)
        self.entrynombre2=ttk.Entry(self.labelframe2, textvariable=self.nombre2, state="readonly")
        self.entrynombre2.grid(column=1, row=1, padx=4, pady=4)
        self.label22=ttk.Label(self.labelframe2, text="Apellidos:")        
        self.label22.grid(column=0, row=2, padx=4, pady=4)
        self.apellidos2=tk.StringVar(self.labelframe2)
        self.entryapellidos2=ttk.Entry(self.labelframe2, textvariable=self.apellidos2, state="readonly")
        self.entryapellidos2.grid(column=1, row=2, padx=4, pady=4)
        self.label3=ttk.Label(self.labelframe2, text="CIF:")        
        self.label3.grid(column=0, row=3, padx=4, pady=4)
        self.cif2=tk.StringVar(self.labelframe2)
        self.entrycif2=ttk.Entry(self.labelframe2, textvariable=self.cif2, state="readonly")
        self.entrycif2.grid(column=1, row=3, padx=4, pady=4)
        self.label42=ttk.Label(self.labelframe2, text="Móvil:")        
        self.label42.grid(column=0, row=4, padx=4, pady=4)
        self.movil2=tk.StringVar(self.labelframe2)
        self.entrymovil2=ttk.Entry(self.labelframe2, textvariable=self.movil2, state="readonly")
        self.entrymovil2.grid(column=1, row=4, padx=4, pady=4)
        self.label5=ttk.Label(self.labelframe2, text="mail:")        
        self.label5.grid(column=0, row=5, padx=4, pady=4)
        self.mail2=tk.StringVar(self.labelframe2)
        self.entrymail2=ttk.Entry(self.labelframe2, textvariable=self.mail2, state="readonly")
        self.entrymail2.grid(column=1, row=5, padx=4, pady=4)
        self.label62=ttk.Label(self.labelframe2, text="Departamento:")        
        self.label62.grid(column=0, row=6, padx=4, pady=4)
        self.departamento2=tk.StringVar(self.labelframe2)
        self.entrydepartamento2=ttk.Entry(self.labelframe2, textvariable=self.departamento2, state="readonly")
        self.entrydepartamento2.grid(column=1, row=6, padx=4, pady=4)
        self.label72=ttk.Label(self.labelframe2, text="Cargo:")        
        self.label72.grid(column=0, row=7, padx=4, pady=4)
        self.cargo2=tk.StringVar(self.labelframe2)
        self.entrycargo2=ttk.Entry(self.labelframe2, textvariable=self.cargo2, state="readonly")
        self.entrycargo2.grid(column=1, row=7, padx=4, pady=4)
        self.boton12=ttk.Button(self.labelframe2, text="Consultar", command=self.consultar)
        self.boton12.grid(column=1, row=8, padx=4, pady=4)


    def consultar(self):

        print("DNI recibido en consulta:",self.dni2.get())
        datos=(self.dni2.get(), )
        respuesta=self.empleado1.consulta(base_de_datos,datos)
        print("nombre respuesta a consulta:",self.nombre2.get())
        print(respuesta)
        if len(respuesta)>0:
            
            self.dni2.set(respuesta[0][0])
            self.nombre2.set(respuesta[0][1])
            self.apellidos2.set(respuesta[0][2])
            self.cif2.set(respuesta[0][3])
            self.movil2.set(respuesta[0][4])
            self.mail2.set(respuesta[0][5])
            self.departamento2.set(respuesta[0][6])
            self.cargo2.set(respuesta[0][7])
        else:
            self.dni2.set('')
            self.nombre2.set('')
            self.apellidos2.set('')
            self.cif2.set('')
            self.movil2.set('')
            self.mail2.set('')
            self.departamento2.set('')
            self.cargo2.set('')
            mb.showinfo("Información", "No hay  empleados con ese DNI")

    def listado_completo(self):
        self.pagina3 = ttk.Frame(self.cuaderno1)
        self.cuaderno1.add(self.pagina3, text="Listado completo")
        self.labelframe3=ttk.LabelFrame(self.pagina3, text="Empleados")
        self.labelframe3.grid(column=0, row=0, padx=5, pady=10)
                
        self.boton1=ttk.Button(self.labelframe3, text="Listado completo", command=self.listar)
        self.boton1.grid(column=0, row=0, padx=4, pady=4)
        self.scrolledtext1=st.ScrolledText(self.labelframe3, width=30, height=10)
        self.scrolledtext1.grid(column=0,row=1, padx=10, pady=10)

    def listar(self):
        respuesta=self.empleado1.recuperar_todos(base_de_datos)
        self.scrolledtext1.delete("1.0", tk.END)        
        for fila in respuesta:
            self.scrolledtext1.insert(tk.END, "DNI:"+str(fila[0])+"\nNombre:"+fila[1]+"\nApellidos:"+str(fila[2])+
                "\nCIF:"+str(fila[3])+"\nMovil:"+str(fila[4])+"\nMail:"+str(fila[5])+
                "\nDepartamento:"+str(fila[6])+"\nCargo:"+str(fila[7])+"\n\n")


# In[ ]:


def base_popup():

    aplicacion1=FormularioEmpleados()


# In[ ]:


def seleccionar_in_out():
    global sentido
    sentido=int("{}".format(opcion.get()))
    textbox21.delete("1.0","end")
    textbox31.delete("1.0","end")
    print(sentido)
    if sentido==0:
        textbox21.insert('end', directorio_caras_in)
        textbox31.insert('end', directorio_ocr_in)
    else:
        textbox21.insert('end', directorio_caras_out)
        textbox31.insert('end', directorio_ocr_out)


# In[ ]:


window = tk.Tk()
window.title('Ventana Principal')
window.geometry('1600x900')
etiqueta=tk.Label(window, text='AIPIA  PANEL DE CONTROL', bg='blue', font=('Arial', 16))
etiqueta.grid(row=0, column=0, sticky="nsew")
window.iconbitmap("iconos/aipia.ico") #Cambiar el icono
#window.resizable(0,0)  # Bloquear tamaño ventana
window.title("AIPIA  PANEL DE CONTROL")
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=12)
window.columnconfigure(0, weight=1)
global sentido, video_on_off,fotos_on_off, cola_whastapp
sentido=0
video_on_off=0
fotos_on_off=0
cola_whastapp=Cola()
#Barra Menu        
        
menu = Menu(window)
fichero = Menu(menu)
fichero.add_command(label='Abrir', command = abrir)
fichero.add_command(label='Crear', command = crear)
fichero.add_command(label='Graba', command = grabar)
fichero.add_command(label='Carpeta', command = carpeta_base)
fichero.add_command(label='Salir', command = salir)
menu.add_cascade(label='Fichero', menu=fichero)

configurar = Menu(menu)
configurar.add_command(label='Camara Entrada', command = cam_in)
configurar.add_command(label='Camara Salida', command = cam_out)
configurar.add_command(label='Directorio Rostros registrados', command =dir_rostros_reg)
configurar.add_command(label='Carpeta Rostros Entrada', command = dir_rostros_in)
configurar.add_command(label='Carpeta Rostros Salida', command = dir_rostros_out)
menu.add_cascade(label='Configurar', menu=configurar)

imprimir = Menu(menu)
imprimir.add_command(label='Seleccionar Impresora',command = impresora_local)
imprimir.add_command(label='Imprime Listado',command = imprimecuadro)
menu.add_cascade(label='Imprimir', menu=imprimir)

ayuda = Menu(menu)
ayuda.add_command(label='Sobre AIPIA',command = sobre)
menu.add_cascade(label='Ayuda', menu=ayuda)
window.config(menu=menu)

#Marcos

# Crea el marco de la segunda capa
frame=tk.Frame(window)
frame.grid(row=1, column=0, sticky="nsew")
frame.grid_propagate(0)
# Crea marcos de la segunda capa, que crece en el marco principal
frame_lu = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)
frame_ru = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)
frame_ld = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)
frame_rd = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)
frame_cu = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)
frame_cd = tk.Frame(frame, bg='blue',relief=GROOVE,bd=5)

frame_lu.grid(row=0, column=0, sticky="nsew")
frame_ru.grid(row=0, column=2, sticky="nsew")
frame_cu.grid(row=0, column=1, sticky="nsew")
frame_ld.grid(row=1, column=0, sticky="nsew")
frame_rd.grid(row=1, column=2, sticky="nsew")
frame_cd.grid(row=1, column=1, sticky="nsew")

frame_lu.grid_propagate(0)
frame_ru.grid_propagate(0)
frame_ld.grid_propagate(0)
frame_rd.grid_propagate(0)
frame_cu.grid_propagate(0)
frame_cd.grid_propagate(0)

img1 = ImageTk.PhotoImage(Image.open("iconos/aipia.jpg"))
panelA = tk.Label(frame_lu, image=img1, bg="blue", width=480)
panelA.grid(row=0, column=0, padx=0,pady=0,sticky="nsew")

opcion = IntVar()

rb1=Radiobutton(frame_lu, text="Entrada", variable=opcion, 
            width=9,value=0, command=seleccionar_in_out)
rb1.grid(row=0, column=0, padx=0,pady=0,sticky="ne")

rb2=Radiobutton(frame_lu, text="Salidas ", variable=opcion, 
            width=9,value=1, command=seleccionar_in_out)
rb2.grid(row=0, column=0, padx=0,pady=20,sticky="ne")



textbox10=tk.Text(frame_ru, bg='white',height=5, width=40)
textbox10.grid(row=0,column=0, padx=0,pady=0,sticky="nsew")
textbox10.insert('end', "")

photo20=ImageTk.PhotoImage(file="iconos/on1.png")
photo21=ImageTk.PhotoImage(file="iconos/on2.png")
button20 = tk.Button(frame_lu, image=photo20, width=80, bg='blue',text="Iniciar YoloV2", command=yolo2_on, relief=RAISED)
button20.grid(row=0, column=0, sticky="sw")

button21 = tk.Button(frame_lu, image=photo20, width=80, bg='blue',text="Iniciar Fotos", command=yolo2_fotos_on, relief=RAISED)
button21.grid(row=0, column=0, sticky="nw")

button22 = tk.Button(frame_lu, image=photo20, width=80, bg='blue', text="Video On", command=video_on, relief=RAISED)
button22.grid(row=0, column=0, sticky="se")

button24 = tk.Button(frame_ld, image=photo20, width=80, bg='blue',text="Iniciar Caras=>DNI",command=caras_on, relief=RAISED)
button24.grid(row=0, column=0, sticky="nw")

button26 = tk.Button(frame_ld, image=photo20, width=80, bg='blue',text="Ver Caras On",command=ver_caras_on, relief=RAISED)
button26.grid(row=0, column=3, sticky="ne")


textbox20=tk.Text(frame_ld, bg='white',height=1, width=20,font=("Verdana",7))
textbox20.grid(row=2,column=3, padx=0,pady=0,sticky="new")
textbox20.insert('end', directorio_actual)
textbox21=tk.Text(frame_ld, bg='white',height=1, width=20,font=("Verdana",7))
textbox21.grid(row=2,column=3, padx=0,pady=20,sticky="new")
textbox21.insert('end', directorio_caras_in)
textbox22=tk.Text(frame_ld, bg='white',height=1, width=20,font=("Verdana",10))
textbox22.grid(row=2,column=3, padx=0,pady=40,sticky="new")
textbox22.insert('end', caras_cola)
textbox23=tk.Text(frame_ld, bg='white',height=1, width=20,font=("Verdana",8))
textbox23.grid(row=2,column=3, padx=0,pady=60,sticky="new")
textbox23.insert('end', persona_reg)
textbox24=tk.Text(frame_ld, bg='white',height=10, width=20,font=("Verdana",7))
textbox24.grid(row=2,column=3, padx=0,pady=80,sticky="new")
textbox24.insert('end', "")

label20 = tk.Label(frame_ld, text="Carpeta Actual:", bg="blue",font=("Verdana",10), anchor="e")
label20.grid(row=2,column=2, padx=0,pady=0,sticky="new")
label21 = tk.Label(frame_ld, text="Carpeta Caras:", bg="blue",font=("Verdana",10), anchor="e")
label21.grid(row=2,column=2, padx=0,pady=20,sticky="new")
label22 = tk.Label(frame_ld, text="Caras, Cola:", bg="blue",font=("Verdana",10), anchor="e")
label22.grid(row=2,column=2, padx=0,pady=40,sticky="new")
label23 = tk.Label(frame_ld, text="Persona Identificada:", bg="blue",font=("Verdana",10),anchor="e")
label23.grid(row=2,column=2, padx=0,pady=60,sticky="new")
label24 = tk.Label(frame_ld, text="Información:", bg="blue",font=("Verdana",10),anchor="e")
label24.grid(row=2,column=2, padx=0,pady=80,sticky="new")

img2 = ImageTk.PhotoImage(Image.open("iconos/ud.jpg"))
panelB = tk.Label(frame_ld, image=img2, bg="blue", width=200)
panelB.grid(row=2, column=1, padx=0,pady=0,sticky="nw")

img22 = ImageTk.PhotoImage(Image.open("iconos/DNI.jpg"))
panelBB = tk.Label(frame_ld, image=img22, bg="blue", width=200)
panelBB.grid(row=2, column=0, padx=0,pady=0,sticky="nw")

button30 = tk.Button(frame_rd, text="Tabla Registro",width=20,command=sql_registro,relief=GROOVE)
button30.grid(row=0, column=0, sticky="nw")
button31 = tk.Button(frame_rd, text="Tabla Fichajes",width=20,command=sql_fichajes,relief=RIDGE)
button31.grid(row=0, column=1, sticky="nw")
button32 = tk.Button(frame_rd, text="Limpia Texto",width=20,command=limpiacuadrotexto)
button32.grid(row=0, column=2, sticky="nw")
button33 = Button(frame_rd, text ="Imprimir",width=12, command=imprimecuadro)
button33.grid(row=0, column=3, sticky="nw")
button34 = tk.Button(frame_rd, text="Base Datos",width=12,command=base_popup)
button34.grid(row=0, column=3, sticky="ne")

button35 = tk.Button(frame_rd, text="Iniciar OCR",width=20,command=ocr_on, relief=RAISED)
button35.grid(row=1, column=0, sticky="nw")
button36 = tk.Button(frame_rd, text="Parar OCR",width=20,command=ocr_off, relief=RAISED)
button36.grid(row=1, column=1, sticky="nw")
button37 = tk.Button(frame_rd, text="Ver Placa ON",width=20,command=ver_ocr_on, relief=RAISED)
button37.grid(row=1, column=2, sticky="nw")
button38 = tk.Button(frame_rd, text="Ver Placa OFF",width=20,command=ver_ocr_off, relief=SUNKEN)
button38.grid(row=1, column=3, sticky="nw")


photo44=ImageTk.PhotoImage(file="iconos/wa.png")
button44 = tk.Button(frame_cd, image=photo44, width=80, bg='blue',text="WhastApp",command=whastapp_on, relief=RAISED)
button44.grid(row=0, column=0, sticky="n")

textbox40=tk.Text(frame_cd, bg='white',height=1, width=20,font=("Verdana",7))
textbox40.grid(row=2,column=0, padx=0,pady=0,sticky="new")
textbox40.insert('end', directorio_actual)
textbox42=tk.Text(frame_cd, bg='white',height=1, width=10)
textbox42.grid(row=2,column=0, padx=0,pady=40,sticky="new")
textbox42.insert('end', cola_whastapp.tamano()) 
textbox44=tk.Text(frame_cd, bg='white',height=10, width=20,font=("Verdana",7))
textbox44.grid(row=2,column=0, padx=0,pady=80,sticky="new")
textbox44.insert('end', "")


textbox30=tk.Text(frame_rd, bg='white',height=1, width=10)
textbox30.grid(row=2,column=3, padx=0,pady=0,sticky="new")
textbox30.insert('end', base_de_datos)
textbox31=tk.Text(frame_rd, bg='white',height=1, width=10)
textbox31.grid(row=2,column=3, padx=0,pady=20,sticky="new")
textbox31.insert('end', directorio_ocr_in) #directorio_ocr_in directorio_actual
textbox32=tk.Text(frame_rd, bg='white',height=1, width=10)
textbox32.grid(row=2,column=3, padx=0,pady=40,sticky="new")
textbox32.insert('end', ocr_cola) 
textbox33=tk.Text(frame_rd, bg='white',height=1, width=10)
textbox33.grid(row=2,column=3, padx=0,pady=60,sticky="new")
textbox33.insert('end', placa_reg)
textbox34=tk.Text(frame_rd, bg='white',height=10, width=20,font=("Verdana",7))
textbox34.grid(row=2,column=3, padx=0,pady=80,sticky="new")
textbox34.insert('end', "")

label30 = tk.Label(frame_rd, text="Carpeta Base:", bg="blue",font=("Verdana",10),anchor="e")
label30.grid(row=2,column=2, padx=0,pady=0,sticky="new")
label31 = tk.Label(frame_rd, text="Carpeta OCR IN:", bg="blue",font=("Verdana",10),anchor="e")
label31.grid(row=2,column=2, padx=0,pady=20,sticky="new")
label32 = tk.Label(frame_rd, text="OCR, Cola:", bg="blue",font=("Verdana",10),anchor="e")
label32.grid(row=2,column=2, padx=0,pady=40,sticky="new")
label33 = tk.Label(frame_rd, text="Matrícula Identificada:", bg="blue",font=("Verdana",10),anchor="e")
label33.grid(row=2,column=2, padx=0,pady=60,sticky="new")
label34 = tk.Label(frame_rd, text="Información:", bg="blue",font=("Verdana",10),anchor="e")
label34.grid(row=2,column=2, padx=0,pady=80,sticky="new")

img3 = ImageTk.PhotoImage(Image.open("iconos/ud.jpg"))
panelC = tk.Label(frame_rd, image=img3, bg="blue", width=200)
panelC.grid(row=2, column=1, padx=0,pady=0,sticky="nw")

img33 = ImageTk.PhotoImage(Image.open("iconos/OCR.jpg"))
panelCC = tk.Label(frame_rd, image=img33, bg="blue", width=200)
panelCC.grid(row=2, column=0, padx=0,pady=0,sticky="nw")

frame.rowconfigure(0, weight=1)
frame.rowconfigure(1, weight=1)

frame.columnconfigure(0, weight=4)
frame.columnconfigure(1, weight=1)
frame.columnconfigure(2, weight=4)

frame_lu.rowconfigure(0, weight=1)
frame_lu.columnconfigure(0, weight=1)

frame_ru.rowconfigure(0, weight=1)
frame_ru.columnconfigure(0, weight=1)

frame_cu.rowconfigure(0, weight=1)
frame_cu.columnconfigure(0, weight=1)

frame_ld.rowconfigure(0, weight=1)
frame_ld.rowconfigure(1, weight=1)
frame_ld.rowconfigure(2, weight=4)
 
frame_ld.columnconfigure(0, weight=1)
frame_ld.columnconfigure(1, weight=1)
frame_ld.columnconfigure(2, weight=1)
frame_ld.columnconfigure(3, weight=1)

frame_rd.rowconfigure(0, weight=1)
frame_rd.rowconfigure(1, weight=1)
frame_rd.rowconfigure(2, weight=4)
 
frame_rd.columnconfigure(0, weight=1)
frame_rd.columnconfigure(1, weight=1)
frame_rd.columnconfigure(2, weight=1)
frame_rd.columnconfigure(3, weight=1)

frame_cd.columnconfigure(0, weight=1)

frame_cd.rowconfigure(0, weight=1)
frame_cd.rowconfigure(1, weight=1)
frame_cd.rowconfigure(2, weight=4)

window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




