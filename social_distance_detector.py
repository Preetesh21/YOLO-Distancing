from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time

# base path to YOLO directory
MODEL_PATH = "yolo-coco"
MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 50

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
args = vars(ap.parse_args())

def Setup(yolo):
    global net, ln, LABELS
    weights = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
    config = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])
    labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n") 
    print("[INFO] loading YOLO from disk...") 
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_people(frame, net, ln, personIdx=0):
	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	starttime = time.time()
	layerOutputs = net.forward(ln)
	stoptime = time.time()
	print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime-starttime)))
	boxes = []
	centroids = []
	confidences = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates,
				# centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)
	return results

def finals(vs,writer,inp):
	# loop over the frames from the video stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))
		violate = set()
		if len(results) >= 2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			if(inp):
				for i in range(0, D.shape[0]):
					for j in range(i+1, D.shape[1]):
						if D[i, j] < MIN_DISTANCE:
							violate.add(i)
							violate.add(j)
			elif(inp is False):
				for i in range(0, D.shape[0]):
					for j in range(i, D.shape[1]):
						if D[i, j] < MIN_DISTANCE:
							violate.add(i)
							violate.add(j)
		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)
			if i in violate:
				color = (0, 0, 255)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		text = "Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		if(len(violate)>10):
			text="WARNING!!!"
			cv2.putText(frame, text, (200, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		# show the output frame
		cv2.imshow("Getting Ready", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		# if an output video file path has been supplied and the video
		# writer has not been initialized, do so now
		if args["output"] != "" and writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)

		# if the video writer is not None, write the frame to the output
		# video file
		if writer is not None:
			writer.write(frame)



starttime1 = time.time() 
Setup(MODEL_PATH)   

# initialize the video stream and pointer to output video file
print("[INFO] accessing the input video")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
inp =False
if args['input']:
    inp=True
print('[INFO] Social-Distancing Checking Begins')
finals(vs,writer,inp)
stoptime1 = time.time()
print("Total Time taken for the complete processing is {:.4f} seconds.".format((stoptime1-starttime1)))