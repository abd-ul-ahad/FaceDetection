import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./videos/1.mp4')
# cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)

drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = faceMesh.process(imgRGB)

    if r.multi_face_landmarks:
        for faceMarks in r.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceMarks, mpFaceMesh.FACEMESH_TESSELATION, drawSpecs, drawSpecs)

            for id, lm in enumerate(faceMarks.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (28, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break