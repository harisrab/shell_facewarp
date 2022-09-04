# import necessary packages and functions
import cv2, dlib, os
import mls as mls
import numpy as np
import math
from tqdm import tqdm

from utils import getFaceRect, landmarks2numpy, createSubdiv2D, calculateDelaunayTriangles, insertBoundaryPoints
from utils import getVideoParameters, warpTriangle, getRigidAlignment
from utils import teethMaskCreate, erodeLipMask, getLips, getLipHeight, drawDelaunay
from utils import mainWarpField, copyMouth
from utils import hallucinateControlPoints, getInterEyeDistance

global im_fn, video_fn

# state of the process
OnOFF = 0
maxOnOFF = 1

# create a folder for results, if it doesn't exist yet
os.makedirs("video_generated", exist_ok=True) 

""" Get face and landmark detectors"""
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../common/shape_predictor_68_face_landmarks.dat"  # Landmark model location
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)  


""" A "gatekeeper" function """
def WarpWrapper(image, video):
    # Function that checks the image and video and chooses between  showing the original video, image 
    # or running the main algorithm
    global OnOFF # needed to drop "Process" trackbar to 0-state in the end
    
    video_fn = video
    # Create a VideoCapture object
    cap = cv2.VideoCapture(os.path.join("src_videos", video_fn))

    # Check if camera is opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
  
    # read and process the image
    im = cv2.imread(os.path.join("src_images", image))
    
    if im is None:
        print("Unable to read the photo")
    else:
        # scale the image to have a 600 pixel height
        scaleY = 600./im.shape[0]
        im = cv2.resize(src=im, dsize=None, fx=scaleY, fy=scaleY, interpolation=cv2.INTER_LINEAR )  

    onOFF = 1
    Warp(image, im, video_fn, cap)



""" Main algorithm """
def Warp(im_path, im, video_fn, cap):
    global OnOFF # needed to drop "Process" trackbar to 0-state in the end

    ########## Get the parameters and landmarks of the image #########
    im_height, im_width, im_channels = im.shape
    im_fn = os.path.basename(im_path)

    # detect the face and the landmarks
    newRect = getFaceRect(im, faceDetector)
    landmarks_im = landmarks2numpy(landmarkDetector(im, newRect))
    
    # print("Original LM: ", landmarks_im)
    # print("Original LM.shape: ", landmarks_im.shape)
    
    # landmarks_im = []

    # Pick original facial landmarks that were already detected.
    # with open(im_lm_path, 'r') as f:
        
    #     lines = f.readlines()
    #     pts = []
    #     for i in range(3, 3+68):
    #         line = lines[i]
    #         line = line[:-1].split(' ')
    #         pts += [float(item) for item in line]
    #     pts0 = np.array(pts).reshape((68, 2))

    #     landmarks_im = pts0
        

    # landmarks_vd_np = []
    # vd_lm_path = "C:/Users/admin/Documents/projects/livePortraits/data/pred_fls_simp_audio_embed.txt"

    # with open(vd_lm_path, 'r') as f:
    #     landmarks_vd = f.readlines()
    #     landmarks_vd = [x.strip().split(" ") for x in landmarks_vd]
    #     landmarks_vd = [[float(y) for y in x] for x in landmarks_vd]


    #     # pair 2 consecutive landmarks to form a pair of points
    #     for eachFrame in landmarks_vd:
    #         frame = []
    #         for i in range(0, len(eachFrame), 3):
    #             frame.append([eachFrame[i], eachFrame[i+1], eachFrame[i+2]])

    #         # frame = np.delete(frame, [65, 66], axis=0)
    #         # frame = np.unique(frame, axis=0)
    #         landmarks_vd_np.append(frame)
        
    #     # convert the list of lists to a numpy array
    #     landmarks_vd_np = np.array(landmarks_vd_np)

  
    ###########  Get the parameters of the driving video ##########
    # Obtain default resolutions of the frame (system dependent) and convert from float to integer.
    (time_video, length_video, fps, frame_width, frame_height) = getVideoParameters(cap)

    ############### Create new video ######################
    output_fn = im_fn[:-4] + "_" + video_fn
    out = cv2.VideoWriter(os.path.join("video_generated", output_fn),
                          cv2.VideoWriter_fourcc('M','J','P','G'), fps, (im_width, im_height))

    ############### Initialize the algorithm parameters #################
    frame = [] 
    tform = [] # similarity transformation that alignes video frame to the input image
    srcPoints_frame = []
    numCP = 68 # number of control points
    newRect_frame = []

    # Optical Flow
    points=[]
    pointsPrev=[] 
    pointsDetectedCur=[] 
    pointsDetectedPrev=[]
    eyeDistanceNotCalculated = True
    eyeDistance = 0
    isFirstFrame = True

    ############### Go over frames #################
    count = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=length)

    while True:
    # for i, eachFrame_l in enumerate(landmarks_vd_np):
        ret, frame = cap.read()

        pbar.update(1)

        if ret == False:
            # before breaking the loop drop OnOFF to zero and update the "Process trackbar"
            OnOFF = 0
            # cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
            break  
        else:
            # the orginal video doesn't have information about the frame orientation, so  we need to rotate it
            frame = np.rot90(frame, 3).copy()

        # initialize a new frame for the input image
        im_new = im.copy()          

        ###############    Similarity alignment of the frame #################
        # detect the face (only for the first frame) and landmarks
        if isFirstFrame: 
            newRect_frame = getFaceRect(frame, faceDetector)

            # [1] Pick out facial landmarks for the frame
            landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
            # landmarks_frame_init = eachFrame_l

            # compute the similarity transformation in the first frame
            tform = getRigidAlignment(landmarks_frame_init, landmarks_im)    
        else:
            # [1] Pick out facial landmarks for the frame

            landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
            # landmarks_frame_init = eachFrame_l

            if np.array_equal(tform, []):
                print("ERROR: NO SIMILARITY TRANSFORMATION")

        # Apply similarity transform to the frame
        frame_aligned = np.zeros((im_height, im_width, im_channels), dtype=im.dtype)
        frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))

        # Change the landmarks locations
        landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
        landmarks_frame = cv2.transform(landmarks_frame, tform)
        landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))

        # hallucinate additional control points
        if isFirstFrame: 
            (subdiv_temp, dt_im, landmarks_frame) = hallucinateControlPoints(landmarks_init = landmarks_frame, 
                                                                            im_shape = frame_aligned.shape, 
                                                                            INPUT_DIR="", 
                                                                            performTriangulation = True)
            # number of control points
            numCP = landmarks_frame.shape[0]
        else:
            landmarks_frame = np.concatenate((landmarks_frame, np.zeros((numCP-68,2))), axis=0)

        ############### Optical Flow and Stabilization #######################
        # Convert to grayscale.
        imGray = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2GRAY)

        # prepare data for an optical flow
        if (isFirstFrame==True):
            [pointsPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            [pointsDetectedPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            imGrayPrev = imGray.copy()

        # pointsDetectedCur stores results returned by the facial landmark detector
        # points stores the stabilized landmark points
        points = []
        pointsDetectedCur = []
        [points.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
        [pointsDetectedCur.append((p[0], p[1])) for p in landmarks_frame[68:,:]]

        # Convert to numpy float array
        pointsArr = np.array(points, np.float32)
        pointsPrevArr = np.array(pointsPrev,np.float32)

        # If eye distance is not calculated before
        if eyeDistanceNotCalculated:
            eyeDistance = getInterEyeDistance(landmarks_frame)
            eyeDistanceNotCalculated = False

        dotRadius = 3 if (eyeDistance > 100) else 2
        sigma = eyeDistance * eyeDistance / 400
        s = 2*int(eyeDistance/4)+1

        #  Set up optical flow params
        lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
        pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev,imGray,pointsPrevArr,pointsArr,**lk_params)
        sigma = 100

        # Converting to float and back to list
        points = np.array(pointsArr,np.float32).tolist()   


        # Facial landmark points are the detected landmark and additional control points are tracked landmarks  
        landmarks_frame[68:,:] = pointsArr
        landmarks_frame = landmarks_frame.astype(np.int32)

        # getting ready for the next frame
        imGrayPrev = imGray        
        pointsPrev = points
        pointsDetectedPrev = pointsDetectedCur

        ############### End of Optical Flow and Stabilization #######################

        # save information of the first frame for the future
        if isFirstFrame: 
            # hallucinate additional control points for a still image
            landmarks_list = landmarks_im.copy().tolist()
            for p in landmarks_frame[68:]:
                landmarks_list.append([p[0], p[1]])
            srcPoints = np.array(landmarks_list)
            srcPoints = insertBoundaryPoints(im_width, im_height, srcPoints) 

            lip_height = getLipHeight(landmarks_im)            
            (_, _, maskInnerLips0, _) = teethMaskCreate(im_height, im_width, srcPoints)    
            mouth_area0=maskInnerLips0.sum()/255  

            # get source location on the first frame
            srcPoints_frame = landmarks_frame.copy()
            srcPoints_frame = insertBoundaryPoints(im_width, im_height, srcPoints_frame)  

            # Write the original image into the output file
            out.write(im_new)                  

            # no need in additional wraps for the first frame
            isFirstFrame = False
            continue

        ############### Warp Field #######################               
        dstPoints_frame = landmarks_frame
        dstPoints_frame = insertBoundaryPoints(im_width, im_height, dstPoints_frame)

        # get the new locations of the control points
        dstPoints = dstPoints_frame - srcPoints_frame + srcPoints   

        # get a warp field, smoothen it and warp the image
        im_new = mainWarpField(im,srcPoints,dstPoints,dt_im)       

        ############### Mouth cloning #######################
        # get the lips and teeth mask
        (maskAllLips, hullOuterLipsIndex, maskInnerLips, hullInnerLipsIndex) = teethMaskCreate(im_height, im_width, dstPoints)
        mouth_area = maskInnerLips.sum()/255        

        # erode the outer mask based on lipHeight
        maskAllLipsEroded = erodeLipMask(maskAllLips, lip_height)
        
        # smooth the mask of inner region of the mouth
        maskInnerLips = cv2.GaussianBlur(np.stack((maskInnerLips,maskInnerLips,maskInnerLips), axis=2),(3,3), 10)

        # clone/blend the moth part from 'frame_aligned' if needed (for mouth_area/mouth_area0 > 1)
        im_new = copyMouth(mouth_area, mouth_area0,
                            landmarks_frame, dstPoints,
                            frame_aligned, im_new,
                            maskAllLipsEroded, hullOuterLipsIndex, maskInnerLips)           


        # Write the frame into the file 'output.avi'
        out.write(im_new)

        # Display the resulting frame    
        # cv2.imshow('Live Portrets', im_new)

        onOFF = 1
        continue

    pbar.close()

    # When everything is done, release the video capture and video write objects
    cap.release()
    out.release()
          
