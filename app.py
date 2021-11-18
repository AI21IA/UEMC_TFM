import pywhatkit as kit
import time
import pyautogui as pg
global cola_whastapp


class Cola:
    def __init__(self):
        self.items = []

    def estaVacia(self):
        return self.items == []

    def entra(self, item):
        self.items.insert(0,item)

    def sal(self):
        return self.items.pop()

    def tamano(self):
        return len(self.items)
    
def envia_whastapp(a):
    fecha=time.ctime(time.time()+62)
    print(fecha)
    hora=int(fecha[-13:-11])
    segundo=int(fecha[-6:-5])
    if segundo >30:
        minuto=int(fecha[-10:-8])+1
    else:
        minuto=int(fecha[-10:-8])
    telefono="+34"+ str(a[0])
    #kit.sendwhatmsg(telefono, a[1],hora ,minuto)
    kit.sendwhatmsg_instantly(telefono, a[1], wait_time=5)

    time.sleep(1)
    pg.hotkey('ctrl', 'w')
    
def whastapp_on():
    global cola_whastapp
    while cola_whastapp.tamano() >0:
        a=cola_whastapp.sal()
        print(a)
        envia_whastapp(a)
        
cola_whastapp=Cola()

cola_whastapp.entra([671571316,"Hola"])
                    
whastapp_on()

