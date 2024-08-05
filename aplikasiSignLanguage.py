import cv2
import numpy as np
import tensorflow as tf
import os
import pyttsx3


# Load model yang telah dibuat
model = tf.keras.models.load_model(r'D:\SEMESTER 5\COMPUTER VISION\Project Compvis\model_sign-language_kel5.h5')
model.summary()
data_dir = 'Gesture Image Data'
#mengambil label dari data
labels = sorted(os.listdir(data_dir))
labels[-1] = 'Nothing'
print(labels)

#menggunakan camera untuk real time object detection
cap = cv2.VideoCapture(0)

while(True):
    
    _ , frame = cap.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5) 
    #ROI
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50))
    cv2.imshow('roi', roi)
    

    img = img/255

    #make predication about the current frame
    prediction = model.predict(img.reshape(1,50,50,3))
    char_index = np.argmax(prediction)
    #print(char_index,prediction[0,char_index]*100)

    confidence = round(prediction[0,char_index]*100, 1)
    predicted_char = labels[char_index]

    # # Initialize the engine 
    # engine = pyttsx3.init() 
    # engine.say(predicted_char) 
    # engine.runAndWait()

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0,0,0)
    thickness = 2

    #membuat prediksi
    msg = predicted_char +', Conf: ' +str(confidence)+' %'
    cv2.putText(frame, msg, (90,90), font, fontScale, color, thickness)
    
    cv2.imshow('frame',frame)
    
    #menggunakan 'q' untuk mengeluarkan camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
        