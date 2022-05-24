# import libraries
import torch
import numpy as np
import cv2
import tkinter
from time import time
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk

class Gesture:
    """
    Class implements Yolo5 model to make gesture detection from webcam.
    """

    def __init__(self, capture_index, model_name):
        """
        Initialize the class.
        :param capture_index: Webcam input.
        :param model_name: Model location.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video to be processed frame by frame to make prediction.
        :return: opencv2 video capture object.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('yolov5-master', 'custom', path=model_name, source='local', force_reload=True)        
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def process_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
          
            ret, frame = cap.read()
            assert ret
            
            frame = cv2.resize(frame, (640,640))
            
            start_time = time()
            results = self.process_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('Gesture Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
      
        cap.release()
        cv2.destroyAllWindows()
        
# Create a new object and execute.
detector = Gesture(capture_index=0, model_name='runs/train/Best3-5kelas-100epoch-valid-min.halo.tuhan/weights/best.pt')

# GUI Main Window
root = tkinter.Tk()
root.title("Gestur Bahasa Isyarat Indonesia")
root.geometry('600x550')

# Label
label_root = tkinter.Label(root, text= "Gestur yang tersedia", font=("Helvetica, 15"), anchor='w', width= 50)
label_root.pack()

def window_guide():
     
    # rootlevel object which will
    # be treated as a new window
    window_guide = Toplevel(root)
 
    # sets the title of the
    # rootlevel widget
    window_guide.title("Panduan")
 
    # sets the geometry of rootlevel
    window_guide.geometry("640x200")
 
    # A Label widget to show in rootlevel
    Label(window_guide, text ="Panduan penggunaan aplikasi", font=("Helvetica, 17")).pack(pady=5)

    text1 = "1. Tekan tombol Open Cam untuk membuka webcam\n"
    text2 = "2. Posisikan tangan sehingga dapat ditangkap oleh kamera\n"
    text3 = "3. Berikan gestur bahasa isyarat menghadap kamera\n"
    text4 = "4. Apabila gestur terdeteksi, sistem akan menandai arti gestur tersebut\n"
    text5 = "5. Tekan esc untuk menutup window webcam\n"

    Message(window_guide,
     text = text1+text2+text3+text4+text5, 
     justify="left",
     anchor="w" ,
     width=590,
     font=("Helvetica, 15")).pack()


# Image
img_gestures = Image.open("assets\Gestures.png")
img_gestures_resized = img_gestures.resize((550,444), Image.ANTIALIAS)
img_gestures_converted = ImageTk.PhotoImage(img_gestures_resized)

gestures = tkinter.Label(root, image= img_gestures_converted, width=550, height=444)
gestures.pack()

# Buttons
btn_cam = tkinter.Button(root, text ="Open Cam", command=detector)
btn_cam.pack(side='left', anchor='e', expand=True)

btn_guide = tkinter.Button(root, text ="Panduan", command=window_guide)
btn_guide.pack(side='right', anchor='w', expand=True)

root.mainloop()