from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Used for the detection of the face in a particular frame
face_classifier = cv2.CascadeClassifier(r'C:\Users\KAPIL\Downloads\emotion_detection-master\emotion_detection-master\BE-PROJECT\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\KAPIL\Downloads\emotion_detection-master\emotion_detection-master\BE-PROJECT\Emotion_little_vgg.h5')

class_labels = ['Dissatisfied','Neutral','Satisfied']
feedback_count={'Dissatisfied':0,'Neutral':0,'Satisfied':0}

# To access the video camera
cap = cv2.VideoCapture('http://192.168.1.219:8080/video')



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # Convert from BGR to gray colour
    faces = face_classifier.detectMultiScale(gray,1.3,5) # Used to detect face 

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  #Used to show the rectangle by focussing on face and giving color and thickness
        roi_gray = gray[y:y+h,x:x+w]                      # Crop the face
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)   # Used to resize the image
    

        # Check if there is atleast single face
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0   # Analyze the frame on the webcam
            roi = img_to_array(roi)                # Convert it to array
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]  # Return the maximum value in an array so that it can predict the actual emotion 
            
            
            if label=="Dissatisfied":
                feedback_count["Dissatisfied"]=feedback_count["Dissatisfied"]+1
            elif label=="Satisfied":
                feedback_count["Satisfied"]=feedback_count["Satisfied"]+1
            elif label=="Neutral":
                feedback_count["Neutral"]=feedback_count["Neutral"]+1


            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            #print(feedback_count)
            # Find Key with Max Value
            itemMaxValue = max(feedback_count.items(), key=lambda x : x[1])
 
            #print('Max value in Dict: ', itemMaxValue[1])
            #print('Key With Max value in Dict: ', itemMaxValue[0])
         
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        # cv2.putText(frame,feedback_count["Dissatisfied"],(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Emotion Detector',frame)
    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('Overall People are: ', itemMaxValue[0])
p=(feedback_count["Dissatisfied"]/(feedback_count["Dissatisfied"]+feedback_count["Satisfied"]+feedback_count["Neutral"]))*100
print('The percentage of people Dissatisfied are: ',p)
s=(feedback_count["Satisfied"]/(feedback_count["Satisfied"]+feedback_count["Neutral"]+feedback_count["Dissatisfied"]))*100
print('The percentage of people Satisfied are: ',s)
n=(feedback_count["Neutral"]/(feedback_count["Neutral"]+feedback_count["Satisfied"]+feedback_count["Dissatisfied"]))*100
print('The percentage of people Neutral are: ',n)
cap.release()
cv2.destroyAllWindows()
