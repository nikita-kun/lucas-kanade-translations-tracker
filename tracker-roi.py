import cv2
import numpy as np
import time
import random


img = []

# Load parameters from "parameters.txt" file
try:
	parameters = np.loadtxt('parameters.txt', dtype=int)
except:
	parameters = [10, #maxIterations
				5, #initIterations
				5, #similarityBreakThresh
				20, #similarityThresh
				10000, #dissimilarityThresh
				10, #featureX
				10, #featureY
				300, #convergeThresh
				]

# Update parameters array on trackbar event
def updateTrackbars(x):
	global parameters
	parameters[0] = cv2.getTrackbarPos('maxIterations','settings')
	parameters[1] = cv2.getTrackbarPos('initIterations','settings')
	parameters[2] = cv2.getTrackbarPos('similarityBreakThresh','settings')
	parameters[3] = cv2.getTrackbarPos('similarityThresh','settings')
	parameters[4] = cv2.getTrackbarPos('dissimilarityThresh','settings')
	parameters[5] = cv2.getTrackbarPos('featureX','settings')
	parameters[6] = cv2.getTrackbarPos('featureY','settings')
	parameters[7] = cv2.getTrackbarPos('convergeThresh','settings')


# Image gradient, Sobel filter
def imageGradient(img, kernelSize = 5, derivativeOrder = 1, filterScale = 1, outputType =  cv2.CV_64F):
	
	#sobel filter X
	derivativeOrderX = derivativeOrder
	derivativeOrderY = 0
	gradientImageX = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale) #messed params?

	#sobel filter Y
	derivativeOrderX, derivativeOrderY = derivativeOrderY, derivativeOrderX
	gradientImageY = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale)
	
	return gradientImageX, gradientImageY

# Image gradient, Scharr filter
def imageGradient2(img, filterScale = 1, outputType =  cv2.CV_64F):
	
	gradientImageX = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=1.0/filterScale)
	gradientImageY = cv2.Scharr(img, cv2.CV_32F, 0, 1, scale=1.0/filterScale)
	
	return gradientImageX, gradientImageY


# Positioned feature, described by a template, gradients, inverted Hessian
class Feature:
	x = None
	y = None
	x2 = None
	y2 = None
	width = None
	height = None

	hessianInv = [[0.0, 0.0],
				  [0.0, 0.0]]
	gradientX = None 
	gradientY = None 
	template = None

	dP = None
	e = None

	# Feature initialization
	def __init__(self, x, y, width, height, template, kernelSize = 5, filterScale = 32):
		self.x = x
		self.y = y
		x2 = x+width
		y2 = y+height
		self.width = width
		self.height = height
		self.halfHeight = height/2
		self.halfWidth = width/2
		self.area = self.width * self.height

		# crop a feature template
		self.template =  np.array(cv2.getRectSubPix(template, (width, height), self.center()), dtype=float)

		# padded template for kerneling
		templatePadded = cv2.getRectSubPix(template, (width + kernelSize, height + kernelSize), self.center())
		
		#gradient = imageGradient(templatePadded, kernelSize)
		gradient = imageGradient2(templatePadded, filterScale)
		
		kernelOffset = int(np.floor(kernelSize/2))
		# Crop gradients by kernelOffset on each side
		self.gradientX = gradient[0][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		self.gradientY = gradient[1][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		
		# Calculate Hessian
		try:
			self.hessianInv = self.calculateInvertedHessian()
		except:
			print 'Bad feature'
			return None

		print 'Feature:', self.center(), self.area


	# Center point of the feature
	def center(self, offset = [0.0, 0.0]):
		return (self.x + self.halfWidth + offset[0], self.y + self.halfHeight + offset[1])

	# Right-Bottom point of the feature
	def p2(self):
		return (self.x + self.width, self.y + self.height)

	# Move feature by offset
	def translate(self, offset):
		self.dP = offset
		self.x += offset[0]
		self.y += offset[1]
		return self

	# Calculate Hessian matrix for gradient image window
	def calculateInvertedHessian(self):

		H = [[0.0,0.0],
			 [0.0,0.0]]

		# Sum Hessian
		for y in range(0, self.height):
			for x in range(0, self.width):
				H[0][0] += self.gradientX[y][x]*self.gradientX[y][x]
				H[0][1] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][0] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][1] += self.gradientY[y][x]*self.gradientY[y][x]

		return np.linalg.inv(H)


	# Calculate error image, #T(x) - I(x+p)
	def errorImage(self, image, offset = [0.0, 0.0]):
			
		E = self.template - cv2.getRectSubPix(image, (self.width, self.height), self.center(offset))
		e = np.sum(np.power(E,2))/(self.area)
		return E, e

	# Inverse translations tracker for the feature
	# Returns parameter change a.k.a feature position change.
	# If translate = false, then the position of a feature won't be updated.
	def trackInverseTranslations(self, image, similarityThresh, translate = True):
	
		dP = [0.0, 0.0]
		
		# Calculate error image
		E, e = self.errorImage(image)

		if e < similarityThresh:
			return dP, e

		###########################################################
		# 7. Calculate steepest descent parameter (offset) updates
		# S = sum matrix
		S = [[0.0], [0.0]]

		for y in range(0, self.height):
			for x in range(0, self.width):
				S[0][0] += self.gradientX[y][x] * E[y][x]
				S[1][0] += self.gradientY[y][x] * E[y][x]

		############################################################

		############################################################
		# 8. Calculate new parameters (dP)
		# Multiply inverted hessian by steepest descent parameter updates (S)
		dP = np.squeeze(np.dot(self.hessianInv,S))
		
		if translate is True:
			self.e = e
			self.dP = dP
			self.x += dP[0]
			self.y += dP[1]

		return dP, e

		
# Region of interest. Works for translations of regions.
# Allows some local movement for features up to longestDistance.
# normalize() brings outliers back to an initial position relative to inliers.
class RegionOfInterest:
	
	features = []
	distance = {}
	longestDistance = 100

	# Add feature and record relative distance to other features in the ROI
	# Impleented as a naive N^2 graph for roi of N features
	def addFeature(self, newFeature):
		
		self.features.append(newFeature)

		for f in self.features:
			if f == newFeature:
				continue
			
			c1 = f.center()
			c2 = newFeature.center()

			if (f in self.distance):
				self.distance[f][newFeature] = tuple(np.subtract(c1,c2))
			else:
				self.distance[f] = {newFeature : tuple(np.subtract(c1,c2))}


			if (newFeature in self.distance):
				self.distance[newFeature][f] = tuple(np.subtract(c2,c1))
			else:
				self.distance[newFeature] = {f : tuple(np.subtract(c2,c1))}


	def removeFeature(self, feature):

		for f in self.features:
			if f == feature:
				continue
			del self.distance[f][feature]
		
		if feature in self.distance:
			del self.distance[feature]
		self.features.remove(feature)


	# Find a best inlier by random sampling
	# Return best inlier, set of inliers, set of outliers
	# Greedy function for test purposes
	def anchorOutliers(self, maxIterations = 100):

		random.shuffle(self.features)

		anchorFeature = None
		outliersToAnchor = self.features
		inliersToAnchor = self.features

		for f in self.features:
			
			# Limit the amount of iterations for huge ROI's
			maxIterations-=1
			if maxIterations < 0:
				break

			outliersList = []
			inliersList = []

			if anchorFeature is None:
				anchorFeature = f

			for g in self.features:

				if f == g:
					continue

				# If a feature is far away from its position relative to other features -> outlier, else inlier.
				if g.e > parameters[4] or np.linalg.norm(np.subtract(np.subtract(g.center(), f.center()), self.distance[g][f])) > self.longestDistance:
					outliersList.append(g)
				else:
					inliersList.append(g)

			# Keep the best feature found (also its inliers and outliers)
			if len(outliersToAnchor) > len(outliersList) :
				anchorFeature = f
				outliersToAnchor = outliersList
				inliersToAnchor = inliersList

		return anchorFeature, outliersToAnchor, inliersToAnchor


	# Brings outliers back to an initial position relative to inliers.
	def normalize(self):

		anchorFeature, outliersList, inliersList = self.anchorOutliers()
		print 'OUTLIERS', len(outliersList)

		#reset all outliers
		for g in outliersList:
			if anchorFeature == g:
				continue
			
			center = None	

			#calculate new outlier's center using inliers distance matrix
			for i in inliersList:	
				newCenter = np.add(i.center(), self.distance[g][i])
				newCenter -= [g.halfWidth, g.halfHeight]
				if center is None:
					center = newCenter
				else:
					center = (center + newCenter)/2

			#update outlier's position		
			if center is not None:
				g.x = center[0]
				g.y = center[1]


	#remove all features
	def clear(self):
		while (len(self.features) > 0 ):
			self.removeFeature(self.features[0])


# Iterate tracking for features in ROI
def trackerIterator(roi, img, similarityThresh, maxIterations, iterations, dissimilarityThresh, similarityBreakThresh, convergeThresh):

	for feature in roi.features:
		
		if feature is None:
			continue

		# error before tracking
		e = float("inf")
		# error after tracking and translating
		e2= float("inf")

		# Total max iterations
		mIter = maxIterations

		# Max iterations with small movements
		i = iterations
		
		# tracking loop
		while(True):

			# Out of iterations - stagnation
			if mIter < 0 or i < 0:
				print 'stagnation', e, e2
				break

			mIter-=1

			# track and move a feature 
			dP, e = feature.trackInverseTranslations(img,similarityBreakThresh, True)
			# new error after tracking and moving 
			E2, e2 = feature.errorImage(img, dP)
			# manually update the last error
			feature.e = e2

			# tracked feature movement distance 
			dPnorm = np.sqrt(dP[0]*dP[0]+dP[1]*dP[1])
			
			# stop tracking if too different
			if (e2 > dissimilarityThresh):
				print 'dissimilarityThresh', e2, '>', dissimilarityThresh
				break

			# stop tracking if feature run away
			if dPnorm > roi.longestDistance:
				print 'runaway'
				break

			# stop tracking if already in a very similar place
			if (e2 < similarityThresh):
				print 'similarity', e, e2
				break

			# Decrease stagnation iteration counter if feature has moved less than 0.1 pixels
			if dPnorm <= 0.1:
				i-=1
			else:
				i+=1

			# Stop tracking, if in relatively similar place and almost not moving anymore
			if e2 < convergeThresh and dPnorm <= 0.02:
				print 'converged?', dP, e, e2
				break

	return roi

# Add feature to the global ROI
def featureAdd(x,y,w=None,h=None):
	global img, parameters, roi

	if w is None:
		w = parameters[5]
	if h is None:
		h = parameters[6]
	feature = Feature(x - int(w/2), y - int(h/2), w, h, img)

	if feature is not None:
		roi.addFeature(feature)

# Mouse click handler
# Add a new feature in place of the click
def click(event, x, y, flags, param):
	global roi
	if event == cv2.EVENT_LBUTTONDOWN:
		featureAdd(x,y)


# Windows initialization
cv2.namedWindow('settings', cv2.WINDOW_NORMAL)
cv2.createTrackbar('maxIterations','settings',parameters[0],100, updateTrackbars)
cv2.createTrackbar('initIterations','settings',parameters[1],100, updateTrackbars)
cv2.createTrackbar('similarityBreakThresh','settings',parameters[2],100, updateTrackbars)
cv2.createTrackbar('similarityThresh','settings',parameters[3],100, updateTrackbars)
cv2.createTrackbar('dissimilarityThresh','settings',parameters[4],50000, updateTrackbars)
cv2.createTrackbar('featureX','settings',parameters[5],100, updateTrackbars)
cv2.createTrackbar('featureY','settings',parameters[6],100, updateTrackbars)
cv2.createTrackbar('convergeThresh','settings',parameters[7],2000, updateTrackbars)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image1", click)


# Tracker input source
source = 'camera' # 'video'

# Open video or device depending on source 
if source == 'video':
	cap = cv2.VideoCapture('ground.mp4')
else:
	cap = cv2.VideoCapture(0)


# New and only region of interest
# All features would belong to this translational ROI
roi = RegionOfInterest()


# Get first frame, wait till Q is pressed.
ret, colorImg = cap.read()
img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
cv2.imshow('image1',img)

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
height , width  =  img.shape
out = cv2.VideoWriter('./output/output' + str(time.time()) +'.avi',fourcc , 25, (width,height))

# Wait for a keypress. 
#Time may be used to set up a ROI
cv2.waitKey(0)

# Main loop
# Get a frame, track features
while(True):
	# Capture frame-by-frame
	ret, colorImg = cap.read()
	img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

	# Launch tracker iterator
	roi = trackerIterator(roi, img, parameters[3], parameters[0], parameters[1], parameters[4], parameters[2], parameters[7])
	roi.normalize()

	# copy an image to draw features on it.
	showcaseImg = img.copy()

	# Draw feature rectangles
	for feature in roi.features:
		p2 = feature.p2()
		rectangle = [feature.x, feature.y, p2[0], p2[1]]
		cv2.rectangle(showcaseImg,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,255,255) if feature.e < 400 else (0,0,0),1)
		
	# Show new frame and output it into a video stream
	cv2.imshow('image1',showcaseImg)
	out.write( cv2.cvtColor(showcaseImg,cv2.COLOR_GRAY2RGB))
	
	# Exit on Q
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Remove features on C
	if cv2.waitKey(10) & 0xFF == ord('c'):
		roi.clear()


cap.release()
out.release()

np.savetxt('parameters.txt', parameters, fmt='%d')

cv2.destroyAllWindows()


