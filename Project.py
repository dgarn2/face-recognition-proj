import cv2


from objloader import *
import sys

import numpy as np

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def drawHand(img, points):
    
    for i in points:
        
        img = cv2.circle(img, (int(i[0][0]),int(i[0][1])), radius=1, color=(150, 100, 100), thickness=-1)
        #img = cv2.circle(img, (int(i[0]),int(i[1])), radius=1, color=(150, 100, 100), thickness=-1)


#################



class Tracker:

    def __init__(self):
        self.currentCorners = None
        self.firstFrame = np.zeros((2,4)).astype(np.float32)
        self.previousFrame = None
        self.previousImg = None

    def initialize(self, img, corners):
        # initialize your tracker with the first frame from the sequence and
        # the corresponding corners from the ground truth
        # this function does not return anything
        
        # Initialize tracker for the first frame   
        # Upper left X
        self.firstFrame[0][0] = corners[0][0] 
        # Upper left Y
        self.firstFrame[0][1] = corners[1][0]
        # Upper right X
        self.firstFrame[0][2] = corners[0][1]
        # Upper right Y
        self.firstFrame[0][3] = corners[1][1]
        # Lower right X
        self.firstFrame[1][0] = corners[0][2]
        # Lower right Y
        self.firstFrame[1][1] = corners[1][2]
        # Lower left X
        self.firstFrame[1][2] = corners[0][3]
        # Lower left Y
        self.firstFrame[1][3] = corners[1][3] 
    
        self.previousImg = img.copy()
        self.previousFrame = cv2.cvtColor(self.previousImg, cv2.COLOR_BGR2GRAY)        

    def update(self, img):
        # update your tracker with the current image and return the current corners    
        self.firstFrame = np.reshape(self.firstFrame, (4,1,2))
        lk_params = dict( winSize  = (69,69), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        nextFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.previousFrame, nextFrame, self.firstFrame, None, **lk_params)
        
        self.current_corners = p1[st==1].T
        self.previousFrame = nextFrame.copy()
        
        #print(self.firstFrame,self.current_corners, p1)
        reshaped_p1 = np.reshape(p1, (2,4)) 
        self.firstFrame = reshaped_p1
        
        return self.current_corners


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))





def get_ground_truth(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
    #cv2.imshow("my face", frame)
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        print(x, y, w, h)
        l = [[x,x+w,x+w,x],[y,y,y+h,y+h]]
        return (x+40, y+20, w-80, h-40)
    else:
        return ()
    

    #TODO CALIBRATE CAMERA







def project(points, K, dist, R, t):
    proj_pts, jac = cv2.projectPoints(points, R, t, K, dist)

    return proj_pts


def wh(a):
    #p=[[a[0],a[1]], [a[0]+a[2], a[1]], [a[0]+a[2], a[1]+a[3]], [a[0], a[1]+a[3]]]
    p=[[a[0],a[0]+a[2],a[0]+a[2],a[0]],
    [a[1], a[1], a[1]+a[3], a[1]+a[3]]]
    return p


################################
def main():
    p=0
    trigger = 0
    tcount =0
    ii=0
    i2=0
    uu=0

    f = 0
    ii=0
    objp = np.float64([[-2,-4,0],[2,-4,0],[2,4,0],[-2,4,0]])
    ob = OBJ("hand.obj", swapyz=True)
    obj_points = ob.vertices
    kl=[]

    faces  = ob.faces
    print(faces[0])
    #transforms
    my= 1
    mx= 1.2
    mz = 1.4
    rx = np.array([[1,0,0], [0, np.cos(3.145*mx), -np.sin(3.145*mx)], [0, np.sin(3.145*mx), np.cos(3.145*mx)]])
    ry = np.array([[np.cos(3.145*my),0 , np.sin(3.145*my)], [0,1,0], [-np.sin(3.145)*my, 0 , np.cos(3.145)*my]])
    rz = np.array([[np.cos(3.145*mz), -np.sin(3.145*mz), 0 ],[np.sin(3.145*mz), np.cos(3.145*mz), 0 ], [0,0,1] ])
    scale= np.array([[1.5,0,0],[0,1.5,0],[0,0,1.5]])


    obj_points = np.transpose(obj_points)
    #obj_points =  np.matmul(ry, obj_points)
    #obj_points =  np.matmul(rx, obj_points)
    #obj_points =  np.matmul(rz, obj_points)
    #obj_points =  np.matmul(scale, obj_points)
    obj_points = np.transpose(obj_points)
    org_obj = obj_points.copy()

    K = np.array([[500, 0, 100], [0, 500, 100], [0, 0, 1]]).astype(np.float32)
   # [[484.04669724,   0.,         335.03555101]
   # [  0.,         499.43275743 ,252.48976628]
    #[  0. ,          0. ,          1.        ]]
    
    dist = np.array([[0,  0.00,  0, 0,
  0]])

   


    track = Tracker()
    #track1 = Tracker()
    #track2 = Tracker()
    #track3 = Tracker()
    
    #manual Video select
    #video_capture = cv2.VideoCapture("./ouwt.mp4")
    video_capture = cv2.VideoCapture("./1.webm")
    #video_capture = cv2.VideoCapture("./111.mp4")
    bbox = []

    


    while len(bbox) < 1:
        for i in range(30):
            ret, frame = video_capture.read()
        #print(ret)
        #print(frame)
        bbox = get_ground_truth(frame)
        
    #tracker.init(frame, bbox)

    x = bbox[0]
    y=bbox[1]
    w=bbox[2]    
    h= bbox[3]
    poin = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

    b1 = wh([bbox[0], bbox[1], bbox[2], bbox[3]])  
    #b2 = wh([bbox[0] + (3*bbox[2]/4), bbox[1], bbox[2]/4, bbox[3]/4])
    #b3 = wh([bbox[0] + (3*bbox[2]/4), bbox[1] + (3*bbox[3]/4), bbox[2]/4, bbox[3]/4])
    #b4 = wh([bbox[0] , bbox[1] + (3*bbox[3]/4), bbox[2]/4, bbox[3]/4])
    

    track.initialize(frame,b1)
    #track1.initialize(frame,b2)
    #track2.initialize(frame,b3)
    #track3.initialize(frame,b4)
    
   
    
    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, poin, K, dist,iterationsCount=1000)
    obj_img_points = project(obj_points, K, dist, rvecs, tvecs)

    

    while True:
        f+=1
        ret, frame = video_capture.read()
        if not ret:
            break

        # Update tracker
        #ret, bbox = tracker.update(frame)
        #bbox = track.update(frame)

        b1=track.update(frame)
        #b2=track1.update(frame)
        #b3=track2.update(frame)
        #b4=track3.update(frame)
        
        



        prevbox = bbox
        bbox = np.float32([[b1[0][0],b1[1][0]], [b1[0][1],b1[1][1]], [b1[0][2],b1[1][2]], [b1[0][3],b1[1][3]]])
        
        #bbox = np.float32([[b1[0][0],b1[1][0]], [b2[0][0],b2[1][0]], [b3[0][0],b3[1][0]], [b4[0][0],b4[1][0]], [bm1[0][0],bm1[1][0]], [bm2[0][0],bm2[1][0]], [bm3[0][0],bm3[1][0]]])
        
        #box centroid
        fcenter = [int(np.sum(bbox[:,0] / 4)), int(np.sum(bbox[:,1] / 4))]
        


        #hand flying
        if trigger == 0:
            #ii+=10

            
            if p == 0:
                for i in range(len(obj_img_points)):
                    obj_img_points[i] = np.array([1000,0]) + obj_img_points[i]
                p=1
            
            for i in range(len(obj_img_points)):
                obj_img_points[i] = np.array([-30,0]) + obj_img_points[i]
            
            obj_img_points1 =np.array( [wi[0] for wi in obj_img_points])
            
           
            top_left = (int(obj_img_points1[:,0].min()), int(obj_img_points1[:,1].min()))
            bottom_right = (int(obj_img_points1[:,0].max()), int(obj_img_points1[:,1].max()))
            
            
            
            
            #intercept
            if top_left[0]< fcenter[0] and bottom_right[0] > fcenter[0] and top_left[1]<fcenter[1] and bottom_right[1]> fcenter[1]:
                trigger = 1
                objfinal = obj_img_points.copy()
            
            #off screne
            elif bottom_right[0]<0:
                
                trigger = 0
                
                uu=0
                ii=0
                obj_points = org_obj.copy()
                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, bbox, K, dist,iterationsCount=1000)
                obj_img_points = project(obj_points, K, dist, rvecs, tvecs)
                for i in range(len(obj_img_points)):
                    obj_img_points[i] = np.array([1000,0]) + obj_img_points[i]
                
        
        #hand thrown
        elif  trigger == 2:
            uu +=1
            
            #translation
            #tranz= np.array([[1,0,ii/200],[0,1,0],[0,0,1]])
            tranz= np.array([0.11,-0.5,-0.6])
            
            for i in range(len(obj_points)):  
                obj_points[i] = tranz + obj_points[i]
                
            
            obj_img_points = project(obj_points, K, dist, rvecs, tvecs)

            #reset
            if uu == 30:
                
                trigger = 0
                
                uu=0
                ii=0
                obj_points = org_obj.copy()
                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, bbox, K, dist)
                obj_img_points = project(obj_points, K, dist, rvecs, tvecs)
                for i in range(len(obj_img_points)):
                    obj_img_points[i] = np.array([1000,0]) + obj_img_points[i]
        
        #hand on face
        elif trigger == 1:
            ii+=1
            

            ret1, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, bbox, K, dist)
            obj_img_points = project(obj_points, K, dist, rvecs, tvecs)
            
            #transition anim
            if ii < 20: 
                
                i2+=0.05
                obj_img_points = ((1.0-i2) * objfinal) + (i2 * obj_img_points)

            #motion trigger - adjust comp for sensitivity
            wid = ((bbox[1][0] + bbox[2][0] +bbox[0][0] + bbox[3][0]) / 4)
            wid2 = ((prevbox[1][0] + prevbox[2][0] + prevbox[0][0]+ prevbox[3][0]) / 4)
            if np.abs(wid - wid2) > 4:
                tcount+=1
                if tcount > 40:
                    trigger = 2
                    i2 = 0      #update
                    

        #list faces and draw
        for nm in faces:
                
            kl = ( [obj_img_points[nm[0][0]-1][0],
                       obj_img_points[nm[0][1]-1][0],
                       obj_img_points[nm[0][2]-1][0],
                       obj_img_points[nm[0][3]-1][0]]
                       )
            

            cv2.fillConvexPoly(frame, np.int32(kl) ,(0, 244, 255))
        


        # Draw bounding box around the tracked face
        if ret:
            #cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

            cv2.circle(frame, (int(b1[0][0]), int(b1[1][0])), radius=3, color=(150, 100, 100), thickness=2)
            #cv2.circle(frame, top_left, radius=7, color=(255, 100, 100), thickness=5)
            #cv2.circle(frame, bottom_right, radius=7, color=(255, 100, 100), thickness=5)
            #cv2.circle(frame, fcenter, radius=7, color=(255, 100, 100), thickness=5)
            #TODO PROJECT OBJ POINTS LIKE A3
            
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            #TODO POSSIBLE SELF ERROR CORRECTION

            # while len(bbox) < 1:
            #     ret, frame = video_capture.read()
            #     bbox = get_ground_truth(frame)
            # tracker = cv2.TrackerKCF_create()
            # print(bbox)
            # #bbox = (100, 100, 100, 100)
            # tracker.init(frame, bbox)

        # Display the resulting frame
       
        #out.write(frame)

        cv2.imshow('Face Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  
    
    video_capture.release()
    cv2.destroyAllWindows()

main()