import cv2
from train import Model
from showImage import show_picture


if __name__=='__main__':
    cap=cv2.VideoCapture(0)
    cascade_path="D:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    model=Model()
    model.load()
    while True:
        _, frame=cap.read()
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.imshow('win',frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        faceRect=cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(faceRect)>0:
            print('face detected!')
            for rect in faceRect:
                x ,y=rect[0:2]
                width,height=rect[2:4]
                image=frame[y:y+height,x:x+width]
                result=model.predict(image)
                cv2.imshow("face",image)

                if result==0:
                    print("Not boss")
                else:
                    print("boss")
                    show_picture()

        # k = cv2.waitKey(100)
        # if k == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()
