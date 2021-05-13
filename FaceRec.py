import cv2
import numpy as np
import face_recognition
import os
import sys
from datetime import datetime
 
path = 'FaceDatabase'
cascPath =  "Resources/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

images = []
classNames = []
queque = []

myList = os.listdir(path)

def getFaceParams(img): 

    faces = faceCascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

    
    if len(queque) == 0:
        for (x,y,w,z) in faces: 

            queque.append(x)
            queque.append(y)
            queque.append(w)
            queque.append(z)



            return queque
    else: 
        return queque



def addFace(name): 

    
    cap = cv2.VideoCapture(0)

    if not os.path.isdir("FaceDatabase/{}".format(name)): 

        dataPath = "FaceDatabase/{}".format(name)

    else: 
       
        name = input(str("Name already exists.\nPlease type a new alias: "))
        dataPath = "FaceDatabase/{}".format(name)
    
    print("{0} Folder Created!".format(dataPath))
    os.mkdir(dataPath)
    numPics = 0


    messages = { 0: "Look straight into camera. Press Space To Capture ",
                1:  "Tilt your head up while keeping your head in the frame. Press Space To Capture ",
                2:  "Tilt your head left while keeping your head in the frame.  Press Space To Capture",
                3:  "Tilt head to right while keeping your head in the frame. Press Space To Capture", 
                4:  "ALL Images collected. Press ESC to Exit stage", 
                }

    queque = [] 

    while True: 

        success, img = cap.read()

        success, imgMarked = cap.read()
             
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for(x,y,h,w) in faces: 

            

            x1 = x - 50 
            h1 = h + 50 

            facerotation = {0 : [(x1,y), (x1,y)],
                            1 : [ (x1, y + h1), (x1, y)],
                            2: [ (x1 + w,y + h1), (x1, y + h1)] ,
                            3: [(x1, y + h1), (x1 + w, y + h1)],
                            4: [(x1,y), (x1,y)]
                            }



            if numPics > 4:
                numPics = 0
            cv2.rectangle(imgMarked, (x, y), (x+w, y+h), (0,0, 255), 2)
            cv2.putText(imgMarked, f"{messages[numPics]}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

            cv2.arrowedLine(imgMarked, facerotation[numPics][0], facerotation[numPics][1], (0,255,0), 10)

            cv2.imshow(name,imgMarked)
            
        
        if cv2.waitKey(1) == 32: # Space bar is pressed 
            cv2.imwrite(filename="{0}/{1}{2}.jpg".format(dataPath,name,numPics),img=img)


            numPics =  numPics +  1
            
            print("Image: {0}{1}.jpg Saved!".format(name,numPics))
        
        elif cv2.waitKey(1) == 27: #ESC key pressed or space pressed more than 4 times

           
            cap.release()
            
            cv2.destroyAllWindows() 

            print("Escape hit, closing window.")

            break


def retrieveFaces(): 

    faces = [] 
    faceList = {} 

    print(myList)
    for person in myList:

        try: 

            os.system("rm .DS_Store")

        except FileNotFoundError as e: 
            pass

        for image in os.listdir(f'{path}/{person}') : 
            curImg = cv2.imread(f'{path}/{person}/{image}')
            
            images.append(curImg)

            faceList[person] = images

            classNames.append(os.path.splitext(person)[0])

    return faceList



def newFace(): 

    name = str(input("Type your name: "))
    addFace(name)

     
def findEncodings(images):
    encodeList = []
    index = 0 

    try: 
        for img in images:
           
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print(f"{len(encodeList)}, {classNames[index]} Encoded")
            index = index + 1
        return encodeList
    except: 
        print("Unable to Encode Image [{0}]".format(classNames[index]))
        exit()
 
 
def encodeDatabase(): 
    
    Face_Database = retrieveFaces()
    
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    



def run_Facial_Recognition(img,encodeListKnown ): 
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)



    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
            
    
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
        
        return img
