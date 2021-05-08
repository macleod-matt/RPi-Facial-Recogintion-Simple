import io
import socket
import struct
import time
import picamera
import cv2
import numpy as np 
from FaceRec import newFace, run_Facial_Recognition, encodeDatabase



try: 

    if sys.argv[1].upper() == "A": 
        newFace() 

except IndexError: 
    
    pass



encodeDatabase()

client_socket = socket.socket()

client_socket.connect(('192.168.0.11', 8000))  # ADD IP HERE

# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
    camera = picamera.PiCamera()
    camera.vflip = False
    camera.resolution = (500, 480)
    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    start = time.time()
    stream = io.BytesIO()
        
    
    for foo in camera.capture_continuous(stream, 'jpeg'):
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
        
        data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        frame_Decoded = cv2.imdecode(data, 1) # decode stream to imge for opencv 

        # Run facial Recognition -----------

        
        frame_metaData = run_Facial_Recognition(frame_Decoded)


        # ----------------------------------


        # Format as Bytes Object 
        
        frame_String = cv2.imencode('.jpeg', frame_metaData)[1].tobytes()
        
        stream =  bytes(str(frame_String), 'utf-8')

        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        # Rewind the stream and send the image data over the wire
        stream.seek(0)
        connection.write(stream.read())
        # If we've been capturing for more than 30 seconds, quit
        if time.time() - start > 60:
            break
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()









