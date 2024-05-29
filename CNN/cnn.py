import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from scipy.signal import convolve2d
import cv2

def imgToArr(imgPath:str,title:str,width:int,height:int):
    # Loading the image
    img=load_img(imgPath, target_size=(width, height))
    imgArray=img_to_array(imgPath)/255.0
    # Display the cropped image
    plt.imshow(imgArray)
    plt.title(title)
    plt.show()
    return np.array(imgArray)

def main():
    orgImg=imgToArr("main-qimg-7a9024d04af922aa744813ab616017a6-lq.jpeg","original image",1000,1000)
    croppedImg=imgToArr("main-qimg-7a9024d04af922aa744813ab616017a6-lq2.jpeg","cropped image",100,100)



if(__name__=="__main__"):
    main()    

