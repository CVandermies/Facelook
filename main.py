import numpy as np
import cv2
import faceTools
import moodTools
from PIL import Image

emojis_data = {
    'angry': cv2.imread("./data/emojis/Angry.png"),
    'disgust': cv2.imread("./data/emojis/Poisoned.png"),
    'fear': cv2.imread("./data/emojis/Fearful.png"),
    'happy': cv2.imread("./data/emojis/Happy.png"),
    'sad': cv2.imread("./data/emojis/Crying.png"),
    'surprise': cv2.imread("./data/emojis/Omg.png"),
    'neutral': cv2.imread("./data/emojis/Neutral.png")
}
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
Claudia = moodTools.callModel()

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
                     
            mood = moodTools.predict(Claudia, element[0])
            print(mood)
            (x,y,w,h) = element[1]

            emoji = emojis_data[mood]
            # Check if the tilting has been calculated
            if element[2] is not None:
                emoji = Image.fromarray(emoji)
                emoji = np.array(emoji.rotate(int(-element[2])))

            # Fit the emoji to the exact size of the face
            emoji = faceTools.resize(emoji, target_size=(w, h), to_gray=False)
            frame[y:y+h, x:x+w, :] = emoji
            
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # If the key pressed is "q" (quit)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()