# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from urllib.request import urlopen
import requests

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
  resp = urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, readFlag)
  return image

def get_image_path_from_menu_id(menu_id) :
    image_obj = requests.get('https://sfuqrfho44.execute-api.us-east-1.amazonaws.com/dev/menu/path/' + menu_id)
    image_path = image_obj.json()['menu_path']
    return image_path

menu_ids = ["gus_new-york_pizza", "surf-n-fries", "healthy_food_menu", "cow_boy_menu", "chipotle", 
"the_bittmore", "deepblue", "the_tipsy_boar", "kosher_menu", "shabu_shabu"]

menu_id = menu_ids[9]

image_path = get_image_path_from_menu_id(menu_id)
image = url_to_image(image_path)
orig = image.copy()
(H, W) = image.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (640, 640)

newH = int(H/32)*32
newW = int(W/32)*32

rW = W / float(newW)
rH = H / float(newH)
# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('D:\\UMN\\internships\\Heali\\Phase2\\frozen_east_text_detection.pb')
# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()
# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < 0.5:
			continue
		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

start_end_coordinates = []
box_coordinates = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes :
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    co_ordinates = [startX, startY, endX, endY]
    co_ordinates = {'start_x' : startX, 'start_y' : startY, 'end_x' : endX, 'end_y' : endY}
    start_end_coordinates.append(co_ordinates)
    box_coordinates.append([{'x' : startX, 'y' : startY}, {'x' : startX, 'y' : endY}, {'x' :endX,'y' : startY}, {'x' : endX,'y' : endY}])
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 255), 2)
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)


def post_box_coordinates(start_end_coordinates, box_coordinates, menu_id) :
    data = {'menu_id':menu_id,'start_end_coordinates': start_end_coordinates,
        'box_coordinates':box_coordinates}
    API_ENDPOINT = 'https://sfuqrfho44.execute-api.us-east-1.amazonaws.com/dev/menu/JSON'
    r = requests.post(url = API_ENDPOINT, json = data, headers={'User-Agent': 'Mozilla/5.0'})
    

post_box_coordinates(start_end_coordinates, box_coordinates, menu_id)