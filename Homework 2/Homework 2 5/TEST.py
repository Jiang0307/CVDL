import silence_tensorflow.auto
import tensorflow as tf
import keras
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from tensorflow.keras.models import Model
from keras.applications import ResNet50
from keras.models import Sequential
warnings.filterwarnings("ignore")

SIZE = (224,224)
IMAGE_FOLDER_PATH = os.path.join( os.path.dirname(__file__), "DATA", "PetImages")
MODEL_WITHOUT_AUGMENTATION_PATH = os.path.join(os.path.dirname(__file__) , "DATA" , "model_without_augmentation.h5")
MODEL_WITH_AUGMENTATION_PATH = os.path.join(os.path.dirname(__file__) , "DATA" , "model_with_augmentation.h5")

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
    random.shuffle(test)
    return test

def split_image_label(test):
    test_data = []
    test_label = []
    test_temp = np.asarray(test)
    for image, label in test_temp:
        test_data.append(image)
        test_label.append(label)
        
    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    return test_data , test_label

def prediction_accuracy(test_label, prediction):
    correct = 0
    wrong_index = []
    predicted_label = []
    actual_label = []

    for i in range(len(prediction)):
        predict = np.argmax(prediction[i])
        actual = test_label[i]
        if (predict == actual):
            correct += 1
        else:
            wrong_index.append(i)
            actual_label.append(actual)
            predicted_label.append(predict)

    accuracy = correct / len(test_label) * 100
    accuracy = int(round(accuracy))
    print("accuracy : ", correct / len(test_label) * 100)
    return accuracy

def result_comparison(acc1 , acc2):
    x = ["Before augmentation" , "After augmentation"]
    y = [acc1 , acc2]
    plot = plt.bar(x, y)
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2. , 1.002*height,'%d' % int(height), ha='center', va='bottom')
        
    plt.ylabel("accuracy")
    plt.show()

if __name__ == "__main__": 
    test = load_data()
    test_data , test_label = split_image_label(test)
    print(len(test_data))
    print(len(test_label))
    model_without_augmentation = keras.models.load_model(MODEL_WITHOUT_AUGMENTATION_PATH)    
    model_with_augmentation = keras.models.load_model(MODEL_WITH_AUGMENTATION_PATH)
    
    prediction_without_augmentation = model_without_augmentation.predict(test_data)
    prediction_with_augmentation = model_with_augmentation.predict(test_data)
    
    accuracy_without_augmentation = prediction_accuracy(test_label , prediction_without_augmentation)
    accuracy_with_augmentation = prediction_accuracy(test_label , prediction_with_augmentation)
    
    result_comparison(accuracy_without_augmentation , accuracy_with_augmentation)