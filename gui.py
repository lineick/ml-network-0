from keras.models import load_model
from tkinter import *
import tkinter  as tk
from PIL import Image, ImageDraw
import numpy as np

model = load_model('final_model.h5')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping for model normalization
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg = "white", cursor="cross")
        self.canvas.pack()
        self.label = tk.Label(self, text="draw a number", font=("Roboto", 24))
        self.classify_btn = tk.Button(self, text = "classify", command=self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.image = Image.new("RGB", (200, 200), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (200, 200), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def classify_handwriting(self):
        digit, acc = predict_digit(self.image)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(slf, event):
        slf.x = event.x
        slf.y = event.y
        r=8
        slf.canvas.create_oval(slf.x-r, slf.y-r, slf.x + r, slf.y + r, fill='black')
        slf.draw.ellipse([slf.x-r, slf.y-r, slf.x + r, slf.y + r], fill='white', width=8)

        
app = App()
mainloop()