import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pyttsx3

import numpy as np
#load the trained model to classify sign
from keras.models import load_model
import os
import pickle
from keras.models import model_from_json
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import pickle
from keras.models import model_from_json

# Load the pickle file
with open('traffic_classifier.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the architecture and weights
architecture = model_data['architecture']
weights = model_data['weights']

# Reconstruct the model from the architecture (JSON)
model = model_from_json(architecture)

# Load the weights into the model
model.set_weights(weights)

# Compile the model (needed before using it for predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model loaded successfully from pickle!")


classes = { 
           1:'Speed limit (30 kilometer per hour)',
           2:'Speed limit (50 kilometer per hour)',
           3:'Speed limit (60 kilometer per hour)',
           4:'Speed limit (70 kilometer per hour)',
           5:'Speed limit (80 kilometer per hour)',
           6:'End of speed limit (80 kilometer per hour)',
           7:'Speed limit (100 kilometer per hour)',
           8:'Speed limit (120 kilometer per hour)',
           9:'No passing',
           10:'No passing vehicle over 3.5 tons',
           11:'Right-of-way at intersection',
           12:'Priority road',
           13:'Yield',
           14:'Stop',
           15:'No vehicles',
           16:'Vehicle > 3.5 tons prohibited',
           17:'No entry',
           18:'General caution',
           19:'Dangerous curve left',
           20:'Dangerous curve right',
           21:'Double curve',
           22:'Bumpy road',
           23:'Slippery road',
           24:'Road narrows on the right',
           25:'Road work',
           26:'Traffic signals',
           27:'Pedestrians',
           28:'Children crossing',
           29:'Bicycles crossing',
           30:'Beware of ice/snow',
           31:'Wild animals crossing',
           32:'End speed + passing limits',
           33:'Turn right ahead',
           34:'Turn left ahead',
           35:'Ahead only',
           36:'Go straight or right',
           37:'Go straight or left',
           38:'Keep right',
           39:'Keep left',
           40:'Roundabout mandatory',
           41:'End of no passing',
           42:'End no passing vehicle with a weight greater than 3.5 tons' }
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = image.convert("RGB")
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    pred_prob = model.predict(image)[0]  # Get predicted probabilities for each class
    pred_class = np.argmax(pred_prob)  # Get the index of the class with the highest probability
    sign = classes[pred_class]  # Get the corresponding sign from the classes list
    print(sign)
    engine.say(sign)
    label.configure(foreground='#011638', text=sign)
    engine.runAndWait()

def show_classify_button(file_path):
   classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
   classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
   classify_b.place(relx=0.79,rely=0.46)

def upload_image():
   try:
      file_path=filedialog.askopenfilename()
      uploaded=Image.open(file_path)
      uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
      im=ImageTk.PhotoImage(uploaded)
      sign_image.configure(image=im)
      sign_image.image=im
      label.configure(text='')
      show_classify_button(file_path)
   except:
      pass
engine = pyttsx3.init()
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="check traffic sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
