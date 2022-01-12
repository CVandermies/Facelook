from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import numpy as np

emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def callModel():
    """Reconstruct the model and load the weights of training"""
    class_amount = 7
    model = Sequential([
        Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(pool_size=(5,5), strides=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3,3), strides=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3,3), strides=(2, 2)),
        Flatten(),
        # fully connected
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(class_amount, activation='softmax')
    ])
    model.load_weights('./data/models/weights.h5')

    return model

def predict(model, image):
    """Get a roi (face) and predict emotion using the model"""
    prediction = model.predict(image)
    return emotion_list[np.argmax(prediction)]
    

