import numpy as np
import cv2
import faceTools
import moodTools
import maskTools
from keras.models import load_model as lm
from PIL import Image

emojis_data = {
    'angry': cv2.imread("./data/emojis/Angry.png"),
    'disgust': cv2.imread("./data/emojis/Poisoned.png"),
    'fear': cv2.imread("./data/emojis/Fearful.png"),
    'happy': cv2.imread("./data/emojis/Happy.png"),
    'sad': cv2.imread("./data/emojis/Crying.png"),
    'surprise': cv2.imread("./data/emojis/Omg.png"),
    'neutral': cv2.imread("./data/emojis/Neutral.png"),
    'mask': cv2.imread("./data/emojis/masked.png")
}
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
Claudia = moodTools.callModel()
mask_guesser = lm('./data/models/masked.model')
mask_activated = False
            
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    faces = faceTools.findFaces(frame, face_cascade)
    if faces is not None:
        for element in faces:
            if mask_activated :
                cv2.putText(frame, 'Mask guesser ON', (50, 80), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                masked = maskTools.findMask(mask_guesser, element[3])
                # print(masked)
                if masked == 0: # s'il détecte un masque
                    mood = "mask"
                else:
                    mood = moodTools.predict(Claudia, element[0])
            else:         
                cv2.putText(frame, 'Mask guesser OFF', (50, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                mood = moodTools.predict(Claudia, element[0])
                # print(mood)
            (x,y,w,h) = element[1]

            emoji = emojis_data[mood]
            # Check if the tilting has been calculated
            if element[2] is not None:
                emoji = Image.fromarray(emoji)
                emoji = np.array(emoji.rotate(int(-element[2])))

            # Fit the emoji to the exact size of the face
            emoji = faceTools.resize(emoji, target_size=(w, h), to_gray=False)
            frame[y:y+h, x:x+w, :] = emoji


    cv2.putText(frame, 'Press q to exit', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # If the key pressed is "q" (quit)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    key = cv2.waitKey(10)
    if key == 113:
        break
    elif key == 109 : # si appuie touche m, on active la reconnaissance de masques
        mask_activated = True
    elif key == 110 : # touche n
        mask_activated = False
    else :
        continue

        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()