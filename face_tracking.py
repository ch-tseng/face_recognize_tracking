from mtcnn.mtcnn import MTCNN
import cv2
import imutils

video_file = "/media/sf_VMshare/comein7.mp4"
face_detect = "mtcnn"
displayWidth = 500
min_faceSzie = (30, 30)
#tracker_type = "MEDIANFLOW"  #BOOSTING, CSRT, TLD, MIL, KCF, MEDIANFLOW, MOSSE
tracker_type = "KCF"

if(face_detect=="mtcnn"):
    detector = MTCNN()
elif(face_detect=="dlib"):
    detector = dlib.get_frontal_face_detector()
else:
    detector = cv2.CascadeClassifier(cascade_path)


def get_faces(img):
    faces = []
    if(face_detect=="mtcnn"):
        allfaces = detector.detect_faces(img)
        for face in allfaces:
            print("face", face["box"])
            x = face["box"][0]
            y = face["box"][1]
            w = face["box"][2]
            h = face["box"][3]
            faces.append((int(x),int(y),int(w),int(h)))

    elif(face_detect=="dlib"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faces.append((int(x),int(y),int(w),int(h)))

    else:
        allfaces = detector.detectMultiScale(img, scaleFactor=1.10, minNeighbors=5)
        for face in allfaces:
            (x, y, w, h) = face
            faces.append((int(x),int(y),int(w),int(h)))

    if(len(faces)>0):
        return faces
    else:
        return None

def draw_face(img, bbox, txt):
    fontSize = round(img.shape[0] / 930, 1)
    if(fontSize<0.35): fontSize = 0.35
    boldNum = int(img.shape[0] / 500)
    if(boldNum<1): boldNum = 1

    if(bbox is not None):
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),boldNum)
        print ("draw:", bbox)
        cv2.putText(img, txt, (x, y-(boldNum*3)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (255,0,255), boldNum)

    return img


if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()


VIDEO_IN = cv2.VideoCapture(video_file)
# Get current width of frame
width = VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

hasFrame = True
while hasFrame:
    hasFrame, frame = VIDEO_IN.read()
    if not hasFrame:
        break

    displayImg = frame.copy()
    faceBoxes = get_faces(frame)

    if(faceBoxes is not None):

        face = faceBoxes[0]
        displayImg = draw_face(frame, face, "")
        cv2.imshow("frame", imutils.resize(displayImg, width=displayWidth))
        cv2.waitKey(1)

        ok = tracker.init(frame, face)
        trackStatus = True

        while trackStatus is True:
            hasFrame, frame = VIDEO_IN.read()
            trackStatus, face = tracker.update(frame)

            txtStatus = "tracking"
            displayImg = draw_face(frame, face, txtStatus)
            cv2.imshow("frame", imutils.resize(displayImg, width=displayWidth))
            cv2.waitKey(1)

        facebox = face
        txtStatus = "lost..."

    else:
        facebox = None
        txtStatus = "No face"

    displayImg = draw_face(frame, facebox, txtStatus)
    cv2.imshow("frame", imutils.resize(displayImg, width=displayWidth))
    cv2.waitKey(1)

