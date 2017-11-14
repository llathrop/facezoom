import cv2
import sys

image_path='./'
frame_saver=[]

face_casc_path = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_casc_path)
eye_casc_path = 'haarcascades/haarcascade_eye.xml'
eye_cascade= cv2.CascadeClassifier(eye_casc_path)

try:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1280); 
    video_capture.set(4, 1024); 
except:
    print("couldn't get the camera")
    exit()

ret, frame = video_capture.read()
height,width = frame.shape[:2]
zoom_factor=1.5
zoom_rate=24
use_x=0 
use_y=0
use_h=height
use_w=width
face_detected_lastcycle=False
cycle_counter_length=90
cycle_counter=cycle_counter_length

while True:
    # Capture frame-by-frame
    try:
        ret, frame = video_capture.read()
    except:
        print("couldn't get the camera")
        break # still need to save any captured frames
    orig=frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if cycle_counter%5==0:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        eyes_detected = eye_cascade.detectMultiScale(gray[y:y+h,x:x+w])
        #realign the detection to the original image
        eyes=[]
        for (eye_x,eye_y,eye_w,eye_h) in eyes_detected:
            eyes.append((eye_x+x,eye_y+y,eye_w,eye_h))
            
        # Get the bounding box for each detected eye
        for (eye_x,eye_y,eye_w,eye_h) in eyes:
            # Add a green bounding box to the detections image
            cv2.rectangle(frame, (eye_x,eye_y), (eye_x+eye_w,eye_y+eye_h), (255,0,0), 2)
        
        #zoom image to rect
    if  len(faces)>=1 and w<=width and h <height:
        use_x=(x+use_x*(zoom_rate-1))/zoom_rate
        use_y=(y+use_y*(zoom_rate-1))/zoom_rate
        use_h=(h+use_h*(zoom_rate-1))/zoom_rate
        use_w=(w+use_w*(zoom_rate-1))/zoom_rate
        face_detected_lastcycle=True
        cycle_counter=cycle_counter_length

    else:
        if face_detected_lastcycle==False or abs(use_x-x)>use_w:
            print("noface!")
            zoom_out_rate=zoom_rate*2
            use_x=(0+use_x*(zoom_out_rate-1))/zoom_out_rate
            use_y=(0+use_y*(zoom_out_rate-1))/zoom_out_rate
            use_h=(height+use_h*(zoom_out_rate-1))/zoom_out_rate
            use_w=(width+use_w*(zoom_out_rate-1))/zoom_out_rate
        cycle_counter-=1
        if cycle_counter==0:
            face_detected_lastcycle=False
            cycle_counter=cycle_counter_length

    frame=frame[use_y/zoom_factor:use_y+zoom_factor*use_h,use_x/zoom_factor:use_x+zoom_factor*use_w]

    #print(frame.shape[:2])
    
    # Display the resulting frame
    r = 1700.0 / frame.shape[1]
    dim = (1700, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        frame_saver.append(frame)
        cv2.imwrite(image_path+'test-'+str(count)+'.png',frame)
        print("frame saver")


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

count=0
for frame in frame_saver:
	count+=1
	cv2.imshow('image',frame)
	cv2.waitKey(0)
	
cv2.destroyAllWindows()	
	