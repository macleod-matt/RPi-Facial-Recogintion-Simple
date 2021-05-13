''' 
This program streams a live analytics feed from a raspberry pi to a local machine 

Input Paramters: 

Parameter 1: IP address/ Host name from computer you wish to act as a server 

Parameter 2: To add a new face to the Faceial Recoginiton Database attach the following parameter "A" or "a: 

The live Raspberry Pi video stream is processed through a facial recoginiton pipline,

the resulting video stream is annotated with the persons name, that meta data is then packed into

a scoket where it is passed through a via TCP to port 555 on the local network 

'''


import cv2
import sys
import zmq  
import time
import socket
import imagezmq
import traceback
from time import sleep
from imutils.video import VideoStream
from FaceRec import newFace, run_Facial_Recognition,findEncodings ,retrieveFaces, images
import numpy as np 

Face_Database = retrieveFaces()
    
encodeListKnown = findEncodings(images)

print('Encoding Complete')



try: 
    

    if sys.argv[1] == None: 
        print("You must run the script with an input IP adress or hostname to server")
    	
    if sys.argv[2].upper() == "A": 
        newFace() 

    connect_to = 'tcp://{}:5555'.format(sys.argv[1])
except IndexError: 
    
    pass




def sender_start(connect_to=None):
    sender = imagezmq.ImageSender(connect_to=connect_to)
    sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 2000)  # set a receive timeout
    sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 2000)  # set a send timeout
    return sender

#connect_to = 'tcp://192.168.0.11:5555'
sender = sender_start(connect_to)

rpi_name = socket.gethostname()  # send RPi hostname with each image
picam = VideoStream(usePiCamera=True).start()
time.sleep(3.0)  # allow camera sensor to warm up
time_between_restarts = 5  # number of seconds to sleep between sender restarts
jpeg_quality = 95  # cv2 default quality
try:

    while True: 

        image = picam.read()

        image_Metadata = run_Facial_Recognition(image, encodeListKnown) 
       
        if (np.shape(image_Metadata) == ()): 
            image_Metadata = image

        ret_code, jpg_buffer = cv2.imencode(
            ".jpg", image_Metadata, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        try:
            reply_from_mac = sender.send_jpg(rpi_name, jpg_buffer)

        except (zmq.ZMQError, zmq.ContextTerminated, zmq.Again):

            if 'sender' in locals():

                print('Closing ImageSender.')

                sender.close()

            sleep(time_between_restarts)

            print('Restarting ImageSender.')

            sender = sender_start(connect_to)

except (KeyboardInterrupt, SystemExit):

    pass  # Ctrl-C was pressed to end program
except Exception as ex:
   
    print('Python error with no Exception handler:')
   
    print('Traceback error:', ex)
   
    traceback.print_exc()

finally:
    if 'sender' in locals():
       
        sender.close()
   
    picam.stop()  # stop the camera thread
    sys.exit()
