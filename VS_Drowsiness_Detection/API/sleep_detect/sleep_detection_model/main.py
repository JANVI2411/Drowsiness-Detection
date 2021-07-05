from .base import *
from .helper import gaze_tracking
from .config import *
from . import loader

loader.load_model()

BASE_DIR='videos/'

landmark_csv=pd.DataFrame()
data_dict={}
data_dict['face_status']=[]
data_dict['gaze_line_points']=[] #gaze direction
data_dict['pupil_points']=[]     #left eye, right eye
data_dict['blink_duration']=[]   #blink_duration (current blink-previous link)
data_dict['head_direction']=[]   #left,right,front
data_dict['head_orientation']=[] #yaw roll pitch
data_dict['yawn_status']=[]      #yes/no
data_dict['eye_status']=[]       #sleepy,drowsy,open
data_dict['blink_counter']=[]
data_dict['yawn_counter']=[]
data_dict['gaze_ball_status']=[] #open eye sleeping
data_dict['eye_lip_ratio']=[]


def inference(frame,frame_num,gaze,params,width,height):

    main_start=time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = loader.face_detector_model(gray, 0)   

    if len(rects)!=0:
      gaze.NO_FACE_COUNTER=0 
      start=time.time()
      landmarks = loader.face_landmark_model(gray, rects[0])
      # print(landmarks.part(3).x,landmarks.part(3).y)
      arr=[]
      for pt in range(68):
        arr.append([landmarks.part(pt).x,landmarks.part(pt).y])   
      arr=np.array(arr,dtype='int16') 
      
      gaze.landmarks=np.array(arr,dtype='int16')
      # landmark_csv=landmark_csv.append(pd.DataFrame(arr.reshape(-1).reshape(1,-1)))
      
      del landmarks,arr,rects
      gc.collect();    
      
      head_pose = gaze.get_head_direction()
      
      # pupil_left_coords,origin_left,center_left = gaze.get_pupil_coords('left',gray,height, width)
      # pupil_right_coords,origin_right,center_right = gaze.get_pupil_coords('right',gray, height, width)
      # pupil_loc = gaze.get_pupil_location(pupil_left_coords,center_left,pupil_right_coords,center_right)  
      # frame=gaze.mark_pupil(frame,origin_left,origin_right,pupil_left_coords,pupil_right_coords)
      
      nose_end_point2D,image_points,rotation_vector,translation_vector=gaze.get_gaze(params['model_points'],params['cam_matrix'],params['dist_coeffs'])
      frame=gaze.plot_gaze(frame,image_points,nose_end_point2D)
      #pitch,roll,yaw=gaze.get_head_orientation(model_points,rotation_vector,translation_vector,cam_matrix,dist_coeffs)
      
      ear_ratio,lip_distance=gaze.get_eye_status(frame,frame_num)
      # status=gaze.gaze_ball_detection()

      
      # p1 = [int(image_points[0][0]), int(image_points[0][1])]
      # p2 = [int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])]
      # data_dict['face_status'].append(['yes'])
      # data_dict['gaze_line_points'].append([p1,p2])
      # data_dict['head_direction'].append([head_pose])
      # data_dict['head_orientation'].append([pitch,roll,yaw])
      # data_dict['yawn_status'].append([gaze.yawn_status])
      # data_dict['eye_status'].append([gaze.eye_status])
      # data_dict['gaze_ball_status'].append([status])
      # data_dict['eye_lip_ratio'].append([ear_ratio,lip_distance])
      
      # if c!=0:
      #   if data_dict['eye_status'][c][0]=='blinking' and data_dict['eye_status'][c-1][0]!='blinking':
      #     counter=data_dict['blink_counter'][-1][0]
      #     data_dict['blink_counter'].append([counter+1])
      #     gaze.blink_duration=(c-prev_c)/fps
      #     data_dict['blink_duration'].append([gaze.blink_duration])
      #     prev_c=c
      #   else:
      #     counter=data_dict['blink_counter'][-1][0]
      #     data_dict['blink_counter'].append([counter])
      #     data_dict['blink_duration'].append([gaze.blink_duration])
      # else:
      #   prev_c=0
      #   data_dict['blink_counter'].append([0])
      #   data_dict['blink_duration'].append([0])

      # if c!=0:
      #   if data_dict['yawn_status'][c][0]=='yes' and data_dict['yawn_status'][c-1][0]=='no':
      #     counter=data_dict['yawn_counter'][-1][0]
      #     data_dict['yawn_counter'].append([counter+1])
      #   else:
      #     counter=data_dict['yawn_counter'][-1][0]
      #     data_dict['yawn_counter'].append([counter])
      # else:
      #   data_dict['yawn_counter'].append([0])

      # del gray,nose_end_point2D,image_points,rotation_vector,translation_vector
      # gc.collect();
      
      # w=200
      # h=200
      # x=width-200
      # y=height-200
      # cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), -1)
      # cv2.putText(frame, "Looking at {}-side".format(head_pose), (x+5, y+20),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
      # cv2.line(frame,(x,y+30),(x+w,y+30),(0,0,0),2)
      # cv2.putText(frame, "HEAD:", (x+5, y+50),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
      # cv2.putText(frame, "Yaw: {:.2f}".format(yaw), (x+5, y+70),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
      # cv2.putText(frame, "Roll: {:.2f}".format(roll), (x+5, y+90),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)                
      # cv2.putText(frame, "Pitch: {:.2f}".format(pitch), (x+5, y+110),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
      
      # cv2.line(frame,(x,y+125),(x+w,y+125),(0,0,0),2)
      # cv2.putText(frame, "EYE and YAWN:", (x+5, y+145),
                  # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
      # cv2.putText(frame, "Blink Duration: {:.2f}".format(gaze.blink_duration), (x+5, y+165),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
      # cv2.putText(frame, "Yawning: {}".format(gaze.yawn_status), (x+5, y+185),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
      # if status!='active':
      #   cv2.putText(frame, "{}".format(status), (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

      # print(frame_num, ' : ',head_pose)

    else:
      # landmark_csv=landmark_csv.append(pd.DataFrame([[None]]))
      # data_dict['gaze_line_points'].append([[None,None],[None,None]])
      # data_dict['blink_duration'].append([None])
      # data_dict['head_direction'].append([None])
      # data_dict['head_orientation'].append([None,None,None])
      # data_dict['yawn_status'].append([None])
      # data_dict['eye_status'].append([None])
      # data_dict['gaze_ball_status'].append([None])
      # data_dict['eye_lip_ratio'].append([None,None])
      # counter=data_dict['blink_counter'][-1][0]
      # data_dict['blink_counter'].append([counter])
      # counter=data_dict['yawn_counter'][-1][0] 
      # data_dict['yawn_counter'].append([counter])  

      # face_status='no'
      gaze.NO_FACE_COUNTER+=1
      # print(frame_num, ' : No prediction')
      if gaze.NO_FACE_COUNTER>=gaze.NO_FACE_THRESH:
        # for i in range(gaze.NO_FACE_THRESH):
      #     gaze.gaze_direction=np.append(gaze.gaze_direction,'no_face')
      #     gaze.gaze_direction=np.delete(gaze.gaze_direction,0)
      #     gaze.blink_tracker=np.append(gaze.blink_tracker,'no_face')
      #     gaze.blink_tracker=np.delete(gaze.blink_tracker,0)
      #   face_status='alert'
          cv2.putText(frame, "HEAD COLLAPSE OR UNUSUAL HEAD MOVEMENT", (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      #   print('ALERT : No face detected')
      
      # data_dict['face_status'].append([face_status])

    # del frame
    # gc.collect();
    return frame