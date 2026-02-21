import cv2
import time
import math

LENGTH = 256

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, LENGTH))
    # print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LENGTH))

    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    count = 0
    
    try:
        while True:
            start = time.time() 

            ret, frame = cap.read() # read frame
            print(frame.shape)

            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            cv2.imshow('Camera', frame) # show frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    main()