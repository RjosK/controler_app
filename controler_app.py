import cv2  # Importa OpenCV para manipulación de video e imágenes.
import mediapipe as mp  # Importa MediaPipe para reconocimiento de manos.
import numpy as np  # Importa numpy para operaciones matemáticas y de matriz.
from math import degrees, acos, hypot
import pyautogui #Emulador de teclado y mause
from time import sleep
#--------------------------------------------------------------------------    
def centro_palma(coordenadas_list): #funcion que da el centro de la palma 
    coordenadas = np.array(coordenadas_list)
    centro = np.mean(coordenadas, axis = 0)
    centro = int(centro[0]), int(centro[1])
    return centro   
#-------------------------------------------------------------------------    
# Inicializa los módulos de MediaPipe para dibujar y detectar manos.
mp_hands = mp.solutions.hands
cambiar = 0
#--------------------------------------------------------------------------    
# Activa la cámara para captura de video.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
##vemos que marcas de los dedos usaremos
puntos_pulgar = [1,2,4] #pulgar
puntos_palma = [0,1,2,5,9,13,17]#palma de las manos 
puntos_dedos = [8,12,16,20] #Punto de los dedos (meñique, medio, anular y meñique )
puntos_bajo_dedos = [6,10,14,18] #puntos bajos de los dedos
## Configura el modelo de MediaPipe Hands.  
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands: 
    while True:  
        success, img = cap.read()  # Lee un cuadro de video.
        if not success:  # Si falla la lectura, termina el bucle.
            break  
        #configuramos la imagen para poder leer las manos
        img = cv2.flip(img, 1)  # Voltea la imagen horizontalmente.
        height, width, _ = img.shape  # Obtiene dimensiones del cuadro.
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte a RGB.
        results = hands.process(frame_rgb)  # Procesa la imagen para detectar manos.
        finger_counter = "_" #contador de dedos
#--------------------------------------------------------------------------    
        if results.multi_hand_landmarks: #crea las lineas y marcas de todos los dedos
            coordenadas_pulgar = [] #almacenar las coordenadas del pulgar
            coordenadas_PD = []
            coordenadas_P = []
            coordenadas_PBD = []            
            for hand_landmarks in results.multi_hand_landmarks:
    #--------------------------------------------------------------------------          
                for index in puntos_pulgar:#buscamos las coordenadas actuales de los puntos del pulgar y las almacenamos
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_pulgar.append([x,y])
                for index in puntos_dedos: #buscamos las coordenadas de los puntos de la palma, la punta de los dedos y la parte baja de los dedos
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_PD.append([x,y])                
                for index in puntos_bajo_dedos:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_PBD.append([x,y])                
                for index in puntos_palma:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordenadas_P.append([x,y])
#--------------------------------------------------------------------------    
                #buscamos en la lista cuales son los puntos con los que calcularemos el angulo del pulgar
                p1 = np.array(coordenadas_pulgar[0]) #es el punto 1 del pulgar pero al busarlo en la lista esta en la posicion 0
                p2 = np.array(coordenadas_pulgar[1])#es el punto 2 del pulgar pero al busarlo en la lista esta en la posicion 1
                p3 = np.array(coordenadas_pulgar[2]) #es el punto 4 del pulgar pero al busarlo en la lista esta en la posicion 2
                #calculamos la distancia entre estos puntos para formar un triangulo: p1-p2-p3-p1
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
#--------------------------------------------------------------------------  
                #finalmente se calcula el angulo
                if l1 != 0 and l2 != 0 and l3 != 0: #que todos los valores sean distintos de 0, si no es así, ignora el caso para que no divida x cero y de error
                    cos_angle = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
                    cos_angle = max(-1, min(1, cos_angle))  # Asegurar que esté en el rango [-1, 1]
                    angle = degrees(acos(cos_angle))
                    dedo_pulgar = np.array(False)
                    if angle > 150:
                        dedo_pulgar = np.array(True)
                else:
                    dedo_pulgar = np.array(False)
#--------------------------------------------------------------------------                  
                #indice, medio, anular, meñique
                cx, cy = centro_palma(coordenadas_P)
                #convertimos las listas en arrays
                coordenadas_C = np.array([cx,cy])
                coordenadas_PD = np.array(coordenadas_PD)
                coordenadas_PBD = np.array(coordenadas_PBD)
                #distancias del centro de la palma a la punra de los dedos
                D_centro_PD = np.linalg.norm(coordenadas_C - coordenadas_PD, axis= 1)
                #distancia entre la parte baja de los dedos y el centro
                D_centro_PBD = np.linalg.norm(coordenadas_C - coordenadas_PBD, axis= 1)
                #diferecia entre las distancias
                dif = D_centro_PD - D_centro_PBD
                fingers = dif > 0
                fingers = np.append(dedo_pulgar,fingers)
                finger_counter = str(np.count_nonzero(fingers == True))
#--------------------------------------------------------------------------       
                for id, lm in enumerate(hand_landmarks.landmark):#marca los tres dedos que utlizo para controlar el volumen, pulgar, indice y anular 
                    alto,ancho,color =img.shape
                    cx,cy = int(lm.x*ancho), int(lm.y*alto)
                    if id == 4:
                        x4,y4 =cx,cy
                    elif id == 8:
                        x8,y8 =cx,cy
                    elif id == 12:  # Dedo medio
                        x12, y12 = cx, cy      
                    elif id == 14: #parte media dedo anular
                        x14,y14 = cx, cy 
                    elif id == 6: #parte media dedo indice
                        x6, y6 = cx, cy 
                    elif id == 17: #parte media dedo indice
                        x17, y17 = cx, cy 
#--------------------------------------------------------------------------
            distanciaEntreDedos_indice = hypot(x8-x4,y8-y4)
            distanciaEntreDedos_Medio = hypot(x12 - x4, y12 - y4)
            distanciaPlayPause = hypot(x14 - x4, y14 - y4)
            distanciaEntreDedos_BI = hypot(x6-x4,y6-y4)
            distanciaEntreDedos_Meñique = hypot(x17 - x4, y17 -y4)
  #--------------------------------------------------------------------------          
            if fingers[3] == False and fingers[4] == False: # verifica que estan apagados los otros dedos: da doble funcionalidad para los dedos
                if distanciaEntreDedos_indice < 35 and cambiar == 0: #sube el volumen
                    pyautogui.keyDown('volumeup')
                if distanciaEntreDedos_Medio < 35 and cambiar == 0:# Disminuir volumen con el medio
                    pyautogui.keyDown('volumedown')
                if distanciaPlayPause < 35 and cambiar == 0: # pone pause/play
                    pyautogui.keyDown('playpause')
                    sleep(1)
                    cambiar = 1
                elif distanciaPlayPause >=35:
                    cambiar = 0
            if fingers[1] == False and fingers[2] == False and fingers[3] == False and fingers[4] == False : #agrega otra funcion
                if distanciaEntreDedos_BI < 20 and cambiar == 0: #abre el menu de pestañas
                    pyautogui.keyDown('win')
                    pyautogui.keyDown('tab')
                    cambiar = 1     
                elif distanciaEntreDedos_BI >= 20:
                    pyautogui.keyUp('win')
                    pyautogui.keyUp('tab')  
                    cambiar = 0 
            if distanciaEntreDedos_indice < 20 and cambiar == 0 and fingers[4] == True: #flecha derecha
                pyautogui.keyDown('right')
                cambiar = 1                                       
            elif distanciaEntreDedos_indice >= 20:
                cambiar = 0
            if distanciaEntreDedos_Medio < 20 and cambiar == 0 and fingers[4] == True: #flecha izquierda
                pyautogui.keyDown('left')
                cambiar = 1
            elif distanciaEntreDedos_Medio >=20:
                pyautogui.keyUp("left")
                cambiar = 0
            if distanciaEntreDedos_Meñique < 35 and cambiar == 0 and fingers[4] == True: #enter
                pyautogui.keyDown('enter')
                cambiar = 1
                pyautogui.keyUp('enter')
            elif distanciaEntreDedos_Meñique >= 35:
                cambiar = 0
#--------------------------------------------------------------------------                                                
cap.release()  # Libera la cámara.
cv2.destroyAllWindows()  # Cierra las ventanas.
