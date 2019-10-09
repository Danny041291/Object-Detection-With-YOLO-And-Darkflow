import cv2
import sys
import numpy as np
import threading

from darkflow.net.build import TFNet

def tracking(original_img, predictions):
	image = np.copy(original_img)
	for prediction in predictions:
		top_x = prediction['topleft']['x']
		top_y = prediction['topleft']['y']
		btm_x = prediction['bottomright']['x']
		btm_y = prediction['bottomright']['y']
		confidence = prediction['confidence']
		label = prediction['label'] + " " + str(round(confidence, 3))
		if confidence > 0.3:
			image = cv2.rectangle(image, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
			image = cv2.putText(image, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
	return image
	
def image_prediction(tfnet, source):
	image = cv2.imread(source)
	array = np.array(image, dtype=np.float32)
	results = tfnet.return_predict(array)
	cv2.imshow('image', tracking(image, results))
	cv2.waitKey(0)
	
def video_prediction(tfnet, source):
	cap = cv2.VideoCapture(source)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			frame = np.asarray(frame)
			results = tfnet.return_predict(frame)
			new_frame = tracking(frame, results)
			cv2.imshow('frame',new_frame)
			if cv2.waitKey(25) & 0xFF == ord('q'):
			  break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
	
def main():
	options = {"model": "./cfg/yolo.cfg",
           "load": "./bin/yolo.weights",
           "threshold": 0.1, 
		   "gpu": 1.0}
	tfnet = TFNet(options)
	if sys.argv[1] == 'image':
		image_prediction(tfnet, sys.argv[2] if len(sys.argv) > 2 else './sample_img/sample_dog.jpg')
	elif sys.argv[1] == 'video':
		video_prediction(tfnet, sys.argv[2] if len(sys.argv) > 2 else './sample_video/sample_dog.mp4')
	elif sys.argv[1] == 'camera':
		video_prediction(tfnet, 0)
	
if __name__ == "__main__":
	main()