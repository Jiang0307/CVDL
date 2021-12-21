import silence_tensorflow.auto
import sys
import numpy as np
import cv2
import os
import subprocess
import webbrowser
import time
import keras
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
from tensorflow.keras.models import Model
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from PyQt5 import QtWidgets
from UI_Q5 import UI
warnings.filterwarnings("ignore")

SIZE = (224,224)
OPTIMIZER = "Adam"
MODEL_PATH = os.path.join(os.path.dirname(__file__),r"DATA",r"model_without_augmentation.h5")
IMAGE_FOLDER_PATH = os.path.join( os.path.dirname(__file__), "DATA", "PetImages")

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def plot_to_image (fig):
    fig.canvas.draw()
    width,height = fig.canvas.get_width_height()
    buffer = np.frombuffer( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buffer.shape = (width,height,4)
    buffer = np.roll (buffer,3,axis=2)
    return buffer
    
def count_images(path):
    count = 0
    for file in os.listdir(path):       
        if file.endswith(".jpg"):
            count += 1
    return count

def load_data():
    test = []
    folder = os.scandir(IMAGE_FOLDER_PATH)                       
    for category_folder in folder:
        i = 0
        if category_folder.is_dir(): 
            temp_path = os.path.join(IMAGE_FOLDER_PATH , category_folder)
            count = count_images(temp_path)
            for file in os.listdir(category_folder):       
                if file.endswith(".jpg"):
                    image_path = os.path.join(temp_path , file)
                    image = None
                    try:
                        image = Image.open(image_path).convert("RGB")
                        if image is not None:
                            i += 1
                            label = -1
                            image = cv2.cvtColor(np.asarray(image) , cv2.COLOR_RGB2BGR)
                            image = cv2.resize(image,SIZE)
                            if category_folder.name == "Cat":
                                label = 0
                            elif category_folder.name == "Dog":
                                label = 1
                            temp = [image , label]
                            if (0 <= i) and (i <= count*0.8):
                                pass
                            elif (count*0.8 < i) and (i <= count*0.9):
                                pass
                            else:
                                test.append(temp)
                    except:
                        pass
    return test

def build_model():
    backbone = ResNet50(weights="imagenet" , include_top=False , input_shape=(224,224,3) )
    for layer in backbone.layers:
        layer.trainable = True    
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(2 , activation="softmax")(x)
    model = Model(inputs=backbone.input , outputs=output)
    model.compile(optimizer=OPTIMIZER , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])
    return model

class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.UI_SET(self)
        self.button_1.clicked.connect(self.HW2_5_1)
        self.button_2.clicked.connect(self.HW2_5_2)
        self.button_3.clicked.connect(self.HW2_5_3)
        self.button_4.clicked.connect(self.HW2_5_4)
        self.test = load_data()
        self.class_dict = {0:"Cat",1:"Dog"}
        self.model_5_1 = build_model()
        self.model_5_3 = keras.models.load_model(MODEL_PATH)
 
    def HW2_5_1(self):
        self.model_5_1.summary()
        return

    def HW2_5_2(self):
        path = os.path.join(os.path.dirname(__file__) , "DATA")
        cmd1 = "python -m tensorboard.main --logdir logs_without_augmentation"
        tensorboard_url = "http://localhost:6006/"
        os.chdir(path)
        process = subprocess.Popen(cmd1)
        time.sleep(0.5)
        webbrowser.open(tensorboard_url)
        return  

    def HW2_5_3(self):
        try:
            index = int(self.textEdit.text())
            if index<0 or index>=len(self.test):
                return
            img = self.test[index][0]
            test_img = np.expand_dims(img,axis=0)
            ans = self.model_5_3.predict(test_img)
            
            ans_index = np.argmax(ans[0])
            title = "Class : "+self.class_dict[ans_index]
            
            fig = plt.figure("5-3",figsize=(10,10))
            plt.title(title)
            plt.imshow(img)
            fig = plot_to_image(fig)
            result = cv2.cvtColor(fig, cv2.COLOR_RGBA2RGB)
            plt.close()
            cv2.imshow("5-3",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        except ValueError:
            pass

    def HW2_5_4(self):
        path = os.path.join(os.path.dirname(__file__) , "DATA" , "5-4.jpg")
        img = cv2_imread(path)
        cv2.namedWindow("5-4", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("5-4", 1200 , 800)
        cv2.imshow("5-4",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())