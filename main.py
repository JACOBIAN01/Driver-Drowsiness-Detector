import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
from playsound import playsound
import time

WebCam = cv2.VideoCapture(0)

blink_thresh=0.4
eye_AR_Consec_frame=1
start_time = time.time()
prev_blink_time = None
blink_detected = False
Elapse_Second_Global_Var=0
Global_Time_Diff=0
sleep_counter = 0
sleep_threshold = 30
closed_eye_threshold = 0.2
count =0
Total=0


detector= dlib.get_frontal_face_detector()
lm_model=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(L_start,L_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(R_start,R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def EAR_cal(eye):
    v1=dist.euclidean(eye[1],eye[5])
    v2= dist.euclidean(eye[2], eye[4])
    h1 = dist.euclidean(eye[0], eye[3])
    ear=(v1+v2)/h1
    return ear


while (True):
    _ , frame = WebCam.read()  # Capture The Video
    image_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(image_Gray)

    for face in faces:
        x1= face.left()
        x2=face.right()
        y1=face.top()
        y2=face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(200),2)

        #----------Land Marks----------------#
        shapes = lm_model(image_Gray,face)
        shape=face_utils.shape_to_np(shapes)
       #--------------Eye LandMarks---------#
        Left_Eye=shape[L_start:L_end]
        Right_Eye = shape[R_start:R_end]

        for lpt,rpt in zip(Left_Eye,Right_Eye):
            cv2.circle(frame,lpt,2,(200,200,0),2)
            cv2.circle(frame, rpt, 2, (200, 200, 0),2)

        left_EAR= EAR_cal(Left_Eye)
        right_EAR=EAR_cal(Right_Eye)
        avg=(left_EAR+right_EAR)/2

        elapsed_time = time.time() - start_time
        elapsed_seconds = int(elapsed_time)
        Elapse_Second_Global_Var=elapsed_seconds

        if avg<blink_thresh:

            sleep_counter+=1
            if sleep_counter>=sleep_threshold:
                playsound('C:\\Users\\SUBHADEEP GHORAI\\PycharmProjects\\Eye_Blink_Detection_2\\blink-93025.mp3 ')

                # print("Eyes closed for an extended period. You might be falling asleep.")
                sleep_counter=0



            if   not blink_detected:
                current_time = time.time()
                blink_detected = True

                if prev_blink_time:
                    time_difference = current_time - prev_blink_time
                    Global_Time_Diff=time_difference
                    print(f"Time between blinks: {time_difference:.2f} seconds")
                prev_blink_time = current_time






            cv2.putText(frame, "Blink! / Eyes Closed ", (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1.5 , (0, 200, 0), 2)
            count+=1

        else :
            sleep_counter=0
            blink_detected = False




            if count>=eye_AR_Consec_frame:
                Total+=1
                count=0
        cv2.putText(frame,f'Blinks: {Total}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)





    cv2.putText(frame,f"Time:{Elapse_Second_Global_Var}", (160,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8 ,(0,0, 255), 2)
    cv2.imshow("Eye Blink Detection" , frame)

    if cv2.waitKey(1)& 0xFF ==ord('q'):
        break

WebCam.release()

