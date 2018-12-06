import os
import cv2
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from SegNet.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from pandas_ml import ConfusionMatrix

# Conversion de etiquetas a colores RGB
label2color = {
    0:  (0, 0, 0),
    1:  (70, 70, 70),
    2:  (190, 153, 153),
    3:  (250, 170, 160),
    4:  (220,  20,  60), 
    5:  (153, 153, 153), 
    6:  (153, 153, 153), 
    7:  (128, 64, 128), 
    8:  (244, 35, 232),
    9:  (107, 142, 35), 
    10: (0, 0, 142), 
    11: (102, 102, 156), 
    12: (220, 220, 0)
}

def label2image(img, num_classes=13, height=512, width=512):
    '''
        Toma como entrada una imagen con los valores de las clases 
        en cada pixel y regresa su representacion en colores RGB 
        para poder visualizar la segmentacion semantica.
    '''
    color_mask = np.zeros((height, width, 3), np.uint8())
    for i in range(0, num_classes):
        B, G = np.where(img==i) 
        for r, c in zip(B, G):
            red, green, blue = label2color[i]
            color_mask[r,c,0] = red
            color_mask[r,c,1] = green
            color_mask[r,c,2] = blue
    return color_mask

def one_hot_encode(imgs, height=512, width=512):
    '''
        Realizar la codificacion one-hot de la lista de imagenes
    '''
    one_hot = []
    for i in range(len(imgs)):
        im_class = to_categorical(imgs[i][:,:,0], dtype='uint8')
        while im_class.shape[2] < 13:
            zeros = np.zeros((height, width, 1), dtype='uint8')
            im_class = np.concatenate((im_class, zeros), axis=2)
        one_hot.append(im_class)
    return np.array(one_hot)

def make_mask(img):
    '''
        Entrada: Tensor de dimensiones (height, width, classes)
        Salida: Tensor de dimensiones (height, width)
        Se mapea la salida de la red neuronal a una sola matriz
        la cual contiene las clases de cada pixel. 
    '''
    return np.argmax(img, axis=-1)

def IoU(y_true, y_pred, height=512, width=512, classes=13):
    '''
        Calcula la interseccion sobre la union de las target
        real y la target que fue predecida.
        La funcion asume que las imagenes consisten de un solo canal
        donde estan codificadas las clases de cada pixel.
    '''
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

custom_objects = {
    'MaxPoolingWithArgmax2D':MaxPoolingWithArgmax2D,
    'MaxUnpooling2D': MaxUnpooling2D
}


    
height = 512
width = 512

path = '/home/sjkdm/helvete/code/dataset/'
im_names = os.listdir(path + 'data')
target_names = os.listdir(path + 'targets')
data_name = [(im, target) for im, target in zip(im_names, target_names)]
np.random.shuffle(data_name)

im = cv2.imread(path + 'data/' + data_name[0][0], 1)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (height, width), interpolation=cv2.INTER_NEAREST)

target = cv2.imread(path + 'targets/' + data_name[0][1], 1)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target = cv2.resize(target, (height, width), interpolation=cv2.INTER_NEAREST)    
        
cnn = models.load_model('model/model.h5', custom_objects=custom_objects)
cnn.load_weights('model/weights.h5')
    
target_pred = cnn.predict(im.reshape(1,height,width,3))
print('\n\nSalida segnet = ', target_pred.shape)

mask_pred = make_mask(target_pred)
print('Argmax = ', mask_pred.shape)

label_pred = label2image(mask_pred.reshape(height,width))
print('Imagen RGB = ', label_pred.shape)

label_real = label2image(target[:,:,0])
    
imrealpred =  np.hstack((im, label_real, label_pred))

iou = IoU(target[:,:,0], mask_pred.reshape(height,width))
confmat_real = ConfusionMatrix(target[:,:,0].ravel(), target[:,:,0].ravel())
confmat_pred = ConfusionMatrix(target[:,:,0].ravel(), mask_pred.reshape(height,width).ravel())

print('\n\nIoU = %.4f' % iou)

print("\n\nMatriz de confusion (real):\n%s" % confmat_real)

print("\n\nMatriz de confusion (segnet):\n%s" % confmat_pred)


plt.imshow(imrealpred)
plt.xticks([])
plt.yticks([])
plt.show()
