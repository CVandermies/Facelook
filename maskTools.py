import numpy as np


def findMask(model, face_img):
    """Predict if masked or not"""
    normalized=face_img/255.0
    
    reshaped=np.reshape(normalized,(1,224,224,3))
    reshaped = np.vstack([reshaped])
    prediction = model.predict(reshaped)

    # take the biggest probability (0 = masked, 1 = non)
    value=np.argmax(prediction,axis=1)[0]

    return value