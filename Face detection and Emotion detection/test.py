import cv2
import os,sys
import numpy as np
import faceRecognition as fr


#This module takes images  stored in diskand performs face recognition
if (len(sys.argv)!=3):
	print("Usage: python3 test.py <input_image> <number_of_types_of_images(2/3/4)>")
	exit(0)

test_img=cv2.imread(sys.argv[1])#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

# Comment lines 17-34 after 1st run
faces,faceID = None,None
if sys.argv[2] == "2":
	faces,faceID=fr.getFaces('./trainingImages2/')
elif sys.argv[2] == "3":
	faces,faceID=fr.getFaces('./trainingImages3/')
elif sys.argv[2] == "4":
	faces,faceID=fr.getFaces('./trainingImages4/')
else:
	print("Number of types of images should be 2/3/4")
	exit(0)

face_recognizer=fr.train_classifier(faces,faceID)
if sys.argv[2] == "2":
	face_recognizer.write('./trainingData2.yml')
elif sys.argv[2] == "3":
	face_recognizer.write('./trainingData3.yml')
else :
	face_recognizer.write('./trainingData4.yml')

# Uncomment lines 37-43 for subsequent runs
# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
# if sys.argv[2] == "2":
# 	face_recognizer.read('trainingData2.yml')
# elif sys.argv[2] == "3":
# 	face_recognizer.read('trainingData3.yml')
# else :
# 	face_recognizer.read('trainingData4.yml')

name = None
if sys.argv[2] == "2":
	name={0:"Mahi",1:"Modi"}
elif sys.argv[2] == "3":
	name={0:"Mahi",1:"Modi",2:"Kiran"}
else :
	name={0:"Mahi",1:"Modi",2:"Kiran",3:"Rahul"}

for face in faces_detected:
	(x,y,w,h)=face
	roi_gray=gray_img[y:y+h,x:x+h]
	label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
	fr.draw_rect(test_img,face)
	predicted_name=name[label]
	if(confidence>120):
		continue
	print("Confidence : ",confidence)
	fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(700,700))
cv2.imshow("Face ",resized_img)
cv2.waitKey(10000)#Waits for 10 sec
cv2.destroyAllWindows()