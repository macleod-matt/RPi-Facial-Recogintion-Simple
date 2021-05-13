
import sys
import time
import traceback
import numpy as np
import cv2
from imutils.video import FPS
import imagezmq

try:
    with imagezmq.ImageHub() as image_hub:
        while True:  # receive images until Ctrl-C is pressed
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
            # see opencv docs for info on -1 parameter
            cv2.imshow(sent_from, image)  # display images 1 window per sent_from
            cv2.waitKey(1)
            image_hub.send_reply(b'OK')  # REP reply
except (KeyboardInterrupt, SystemExit):
    pass  # Ctrl-C was pressed to end program; FPS stats computed below
except Exception as ex:
    print('Python error with no Exception handler:')
    print('Traceback error:', ex)
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()  # closes the windows opened by cv2.imshow()
    sys.exit()






