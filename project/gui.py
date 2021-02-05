# region #*Importing Libraray
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import pygame 
import random
pygame.mixer.init()
# endregion

# region #*Load Data
model = load_model("Model/mnist_2.h5", compile=False)
model_2 = load_model("Model/Model_up_2_SGD_25.h5", compile=False)
# endregion


# region #*Prediting Function from Model 
def predict_digit(img):  # Preprocess Image
    img = img.resize((32, 32))

    # Convert to GrayScale
    img = img.convert('L')
    img = np.array(img)

    # Reshaping
    img = img.reshape(1, 32, 32, 1)
    img = img/255.0

    # predicting
    res = model.predict([img])[0]
    print("Model_1:",np.argmax(res), max(res))

    res_2 = model_2.predict([img])[0]
    print("Model_2:",np.argmax(res_2), max(res_2))

    if max(res) < max(res_2):
        print("This is Model_2")
        res = res_2
    else:
        print("This is Model_1")

    return np.argmax(res), max(res)
# endregion

# region #*GUI Application

class App(tk.Tk):
    theLetter = '0'

    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating the interface
        self.canvas = tk.Canvas(self, width=300, height=300, bg='black', cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(
            self, text="Clear", font=("Helvetica", 18), command=self.clear_all)
        self.button_say = tk.Button(self, text="Say!", font=("Helvetica",18), command=self.play)
        self.geometry("600x400")

        #Structuring in Grid
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=5)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_say.grid(row=1, column=1, pady=5,padx=15)

        # Motin and Strat position
        self.canvas.bind("<B1-Motion>", self.draw_lines)
# endregion

# region #*Clear button Actin
    def clear_all(self):
        self.canvas.delete("all")
# endregion

# region #*Handwritting detection Function
    def classify_handwriting(self):
        # Getting Canvas ID
        HWND = self.canvas.winfo_id()
        # Getting Coordinate of the ID canvas
        rect = win32gui.GetWindowRect(HWND)
        # insert it in im
        im = ImageGrab.grab(rect)
        word_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G',
                     17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

        #!Prediction Function Call
        digit, acc = predict_digit(im)
        print(word_dict[digit]+","+ str(int(acc*100))+"%")
        if (word_dict[digit] == self.theLetter):
            self.label.configure(text="Correct!")
        else:
            self.label.configure(text="Incorrect!")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 5
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='white', outline='white')
# endregion

#region #*Random Audio
    def play(self):
        word_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G',
                     17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

        rand = random.choice(word_dict)
        self.theLetter = rand
        pygame.mixer.music.load("audio/"+rand+".mp3")
        pygame.mixer.music.play(loops=0)

app = App()
mainloop()
