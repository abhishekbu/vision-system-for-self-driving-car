# Importing all the libraries
import numpy as np
import cv2
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1";
import keras
from tqdm import tqdm

# Loading the model
model = keras.models.load_model('lane_detection.h5')
model.summary()

# Loading the input video
input_video = "test_files/test_video.mp4"
output_video = "test_files/test_video_out.avi"

# Class for averaging
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

# Takes input image and return the result output
def road_lines(image):
    # Getting the image dimensions
    width, height, depth = image.shape
    
    # Resizing the image
    small_img = cv2.resize(image, (160, 80))
    
    # Adding extra dimension
    small_img = small_img[None,:,:,:]
    
    # Getting the prediction from the model
    prediction = model.predict(small_img)[0] * 255
    
    # Appending the result to the recent_fit
    lanes.recent_fit.append(prediction)
    
    # Only using last ten for average
    if len(lanes.recent_fit) > 10:
        lanes.recent_fit = lanes.recent_fit[1:]
    
    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Creating the blanks of image dimension dimension
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    
    # Stacking the result into green channel with red and yellow channels being zero
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Resizing the result into original input image dimension
    lane_image = cv2.resize(lane_drawn, (height,width))

    # Converting RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Adding the input image with the result 
    result = cv2.addWeighted(image, 1, lane_image, 1, 0, dtype=8)

    # Returning the result image
    return result

# Getting the details of the input image
cap = cv2.VideoCapture(input_video) # Reading the input video
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Reading the frame count
fourcc = cv2.VideoWriter_fourcc(*'XVID') # Video writer for saving the video
fps = int(cap.get(cv2.CAP_PROP_FPS)) # Getting the fps in the video
width, height = cap.get(3), cap.get(4) # Getting the width and height of the video
out = cv2.VideoWriter(output_video, fourcc, 30, (int(width), int(height)))  # 

lanes = Lanes()


for i in tqdm(range(length)):
    try:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = road_lines(frame)
        out.write(result)
        cv2.imshow("Output", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

cap.release()
out.release()
cv2.destroyAllWindows()

