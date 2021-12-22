import silence_tensorflow.auto
import tensorflow as tf
import keras
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from tensorflow.keras.models import Model
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping , TensorBoard
from tensorflow.python.keras.utils.data_utils import validate_file
warnings.filterwarnings("ignore")

BATCH_SIZE = 8
LEARNING_RATE = 0.00001
OPTIMIZER = "Adam"
SIZE = (224,224)
IMAGE_FOLDER_PATH = os.path.join( os.path.dirname(__file__), "DATA", "PetImages")

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def count_images(path):
    count = 0
    for file in os.listdir(path):       
        if file.endswith(".jpg"):
            count += 1
    return count

def load_data():
    train = []
    validation = []
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
                                train.append(temp)
                            elif (count*0.8 < i) and (i <= count*0.9):
                                validation.append(temp)
                            else:
                                test.append(temp)
                    except:
                        pass
    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)    
    return train,validation,test

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

def data_augmentation(train):
    augmentation_data = []
    augmentation_label = []
    train_temp = np.asarray(train)
    for i in range(len(train_temp)):
        img = train_temp[i][0] #TRAIN[i][1]是label
        label = train_temp[i][1]
        # original
        augmentation_data.append(img)
        augmentation_label.append(label)
        # horizontal flip
        h_flip = cv2.flip(img,1)
        augmentation_data.append(h_flip)
        augmentation_label.append(label)
        # brighter
        t1 = random.uniform(0,0.2)
        bright = tf.image.adjust_brightness(img,t1)
        augmentation_data.append(bright)
        augmentation_label.append(label)
        """
        # darker
        t2 = random.uniform(0,0.2)
        dark = tf.image.adjust_brightness(img,t2*-1)
        augmentation_data.append(dark)
        augmentation_label.append(label)
        """
    return augmentation_data , augmentation_label

def split_image_label(train , validation , test):
    train_data = []
    train_label = []
    validation_data = []
    validation_label = []
    test_data = []
    test_label = []
    train_temp = np.asarray(train)
    validation_temp = np.asarray(validation)
    test_temp = np.asarray(test)
    for image, label in train_temp:
        train_data.append(image)
        train_label.append(label)
    for image, label in validation_temp:
        validation_data.append(image)
        validation_label.append(label)
    for image, label in test_temp:
        test_data.append(image)
        test_label.append(label)

    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    validation_data = np.asarray(validation_data)
    validation_label = np.asarray(validation_label)
    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    
    return train_data , train_label , validation_data , validation_label , test_data , test_label

def training(model , train_data , train_label , validation_data , validation_label , augmented):
    if augmented == True:
        TensorBoard_path = os.path.join( os.path.dirname(__file__), "DATA" , "logs_with_augmentation")
        TensorBoard_callback = TensorBoard(log_dir=TensorBoard_path)
        model.fit(train_data , train_label , batch_size=BATCH_SIZE , shuffle=True , epochs=25 , validation_data=(validation_data, validation_label) , callbacks=[TensorBoard_callback] )
        save_path = os.path.join(os.path.dirname(__file__) , "DATA" , "model_with_augmentation.h5")
        model.save(save_path)
    else:
        TensorBoard_path = os.path.join( os.path.dirname(__file__), "DATA" , "logs_without_augmentation")
        TensorBoard_callback = TensorBoard(log_dir=TensorBoard_path)
        model.fit(train_data , train_label , batch_size=BATCH_SIZE , shuffle=True , epochs=25 , validation_data=(validation_data, validation_label) , callbacks=[TensorBoard_callback] )
        save_path = os.path.join(os.path.dirname(__file__) , "DATA" , "model_without_augmentation.h5")
        model.save(save_path)

    return

if __name__ == "__main__": 
    (train , validation , test) = load_data()
    (train_data , train_label , validation_data , validation_label , test_data,test_label) = split_image_label(train,validation,test)
    (train_data_augmented , train_label_augmented) = data_augmentation(train)
    
    model_without_augmentation = build_model()
    training(model_without_augmentation , train_data , train_label , validation_data , validation_label , False)
    
    model_with_augmentation = build_model()
    training(model_with_augmentation , train_data_augmented , train_label_augmented , validation_data , validation_label , True)