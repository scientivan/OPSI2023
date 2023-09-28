from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import os
import numpy as np
from tensorflow import keras
import cv2

model  = keras.models.load_model("C:/Users/pc/Documents/pythonProjects/OPSI2023/Le-Net dengan augmentasi dan hyperparameter/lenet5augmented.h5")

root = Tk()
root.title("Deteksi Kanker Paru berbasi CT-Scan / Dien Muhammad Scientivan Kurniapramono")
root.geometry("500x500")
IMAGE_SIZE = (100,100)
jenis_kanker = ["Adenocarcinoma","Kanker Ganas","Kanker Jinak", "Sehat", "Sel Besar Karsinoma", "Sel Besar Skuamous"]

def showImage():
    path_ = filedialog.askopenfilename(initialdir=os.getcwd(),title="Unggah foto CT-Scan Anda",filetypes=(("JPG file","*.jpg"),("JPEG file","*.jpeg"),("PNG file", "*.png")))
    img = Image.open(path_)
    imagee = keras.preprocessing.image.load_img(path=path_,target_size=(100,100))
    input_arr = keras.preprocessing.image.img_to_array(imagee)
    images = np.reshape(input_arr,newshape=(1,100,100,3))
    images = np.array(images)
    prediction = model.predict(images)
    result.configure(text=jenis_kanker[np.argmax(prediction[0])])
    print(jenis_kanker[np.argmax(prediction[0])])
    img.thumbnail((350,350))
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img
    



print()
frame = Frame(root)
frame.pack(side=BOTTOM,padx=15,pady=15)

label = Label(root)
label.pack()

button = Button(frame,text="Pilih Foto",command=showImage)
button.pack(side=LEFT)

result = Label(root)
result.pack(side=TOP)

root.mainloop()


