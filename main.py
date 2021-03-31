# This is a sample Python script.

import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('ImagesBasic/passport right photo jpeg.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('ImagesBasic/MG_7032.jpg')
img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(img)[0]
encode_face =face_recognition.face_encodings(img)[0]
cv2.rectangle(img, (face_loc[3],face_loc[0],face_loc[1],face_loc[2]),(255,0,255),2)

face_loc_test = face_recognition.face_locations(img)[0]
encode_face_test =face_recognition.face_encodings(img)[0]
cv2.rectangle(img_test, (face_loc[3],face_loc[0],face_loc[1],face_loc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encode_face],encode_face_test)
face_distance = face_recognition.face_distance([encode_face],encode_face_test)
print(results ,face_distance)
cv2.putText(img_test,f'{results} {round(face_distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('swapnil 2', img_test)
cv2.imshow('swapnil', img)
cv2.waitKey(0)

