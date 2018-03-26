import cv2
import numpy as np
import time
import random


img = []


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


def imageGradient(img, kernelSize = 5, derivativeOrder = 1, filterScale = 1, outputType =  cv2.CV_64F):
	
	#sobel filter X
	derivativeOrderX = derivativeOrder
	derivativeOrderY = 0
	gradientImageX = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale) #messed params?

	#sobel filter Y
	derivativeOrderX, derivativeOrderY = derivativeOrderY, derivativeOrderX
	gradientImageY = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale)
	
	return gradientImageX, gradientImageY

def imageGradient2(img, filterScale = 1, outputType =  cv2.CV_64F):
	

	gradientImageX = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=1.0/filterScale)
	gradientImageY = cv2.Scharr(img, cv2.CV_32F, 0, 1, scale=1.0/filterScale)
	
	return gradientImageX, gradientImageY


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

		self.template = np.array(template[y:y2,x:x2], dtype=float)

		templatePadded = cv2.getRectSubPix(template, (width + kernelSize, height + kernelSize), self.center())
		#gradient = imageGradient(templatePadded, kernelSize)
		gradient = imageGradient2(templatePadded, filterScale)
		kernelOffset = int(np.floor(kernelSize/2))
		self.gradientX = gradient[0][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		self.gradientY = gradient[1][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		
		try:
			self.hessianInv = self.calculateInvertedHessian()
		except:
			print 'Bad feature'
			return None

		print 'Feature:', self.center(), self.area


	def center(self, offset = [0.0, 0.0]):
		return (self.x + self.halfWidth + offset[0], self.y + self.halfHeight + offset[1])

	def p2(self):
		return (self.x + self.width, self.y + self.height)

	def translate(self, offset):
		self.dP = offset
		self.x += offset[0]
		self.y += offset[1]
		return self

	def calculateInvertedHessian(self):

		###########################################################
		# 6. Calculate Hessian matrix for gradient image window
		H = [[0.0,0.0],
			 [0.0,0.0]]

		# Sum Hessian
		for y in range(0, self.height):
			for x in range(0, self.width):
				H[0][0] += self.gradientX[y][x]*self.gradientX[y][x]
				H[0][1] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][0] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][1] += self.gradientY[y][x]*self.gradientY[y][x]
		###########################################################

		return np.linalg.inv(H)


	
	def errorImage(self, image, offset = [0.0, 0.0]):
		#T(x) - I(x+p)	
		E = self.template - cv2.getRectSubPix(image, (self.width, self.height), self.center(offset))
		e = np.sum(np.power(E,2))/(self.area)
		return E, e

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
		#print "\n S: \n" , S, '\n\n'
		############################################################

		############################################################
		# 8. Calculate new parameters (dP)
		# Multiply inverted hessian by steepest descent parameter updates (S)
		dP = np.squeeze(np.dot(self.hessianInv,S))
		#print '\n dP: \n', dP, '\n\n'
		
		if translate is True:
			self.e = e
			self.dP = dP
			self.x += dP[0]
			self.y += dP[1]

		return dP, e

		
class RegionOfInterest:
	
	features = []
	distance = {}
	longestDistance = 100

	#add feature and record relative distance to other features in the ROI
	def addFeature(self, newFeature):
		
		self.features.append(newFeature)

		for f in self.features:
			if f == newFeature:
				continue
			
			c1 = f.center()
			c2 = newFeature.center()

			#d = np.linalg.norm(np.squeeze(f.center()) - np.squeeze(newFeature.center()))
			
			if (f in self.distance):
				self.distance[f][newFeature] = tuple(np.subtract(c1,c2))
			else:
				self.distance[f] = {newFeature : tuple(np.subtract(c1,c2))}


			if (newFeature in self.distance):
				self.distance[newFeature][f] = tuple(np.subtract(c2,c1))
			else:
				self.distance[newFeature] = {f : tuple(np.subtract(c2,c1))}

			#self.longestDistance = max(np.linalg.norm(self.distance[newFeature][f]), self.longestDistance)


	def removeFeature(self, feature):

		for f in self.features:
			if f == feature:
				continue
			del self.distance[f][feature]
		
		if feature in self.distance:
			del self.distance[feature]
		self.features.remove(feature)

	#obsolete?
	def anchorOutliers(self, maxIterations = 100):

		random.shuffle(self.features)

		anchorFeature = None
		outliersToAnchor = self.features
		inliersToAnchor = self.features

		for f in self.features:
			
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
				if g.e > parameters[4] or np.linalg.norm(np.subtract(np.subtract(g.center(), f.center()), self.distance[g][f])) > self.longestDistance:
					outliersList.append(g)
				else:
					inliersList.append(g)

			if len(outliersToAnchor) > len(outliersList) :#and f.e < anchorFeature.e:
				anchorFeature = f
				outliersToAnchor = outliersList
				inliersToAnchor = inliersList

		return anchorFeature, outliersToAnchor, inliersToAnchor

	def normalize(self):
		#take a random node
		anchorFeature, outliersList, inliersList = self.anchorOutliers()
		print 'OUTLIERS', len(outliersList)

		for g in outliersList:
			if anchorFeature == g:
				continue
			
			center = None	
			for i in inliersList:	
				newCenter = np.add(i.center(), self.distance[g][i])
				newCenter[0] -= g.halfWidth
				newCenter[1] -= g.halfHeight
				if center is None:
					center = newCenter
				else:
					center[0] = ((center[0]+newCenter[0])/2)# + g.x)/2
					center[1] = ((center[1]+newCenter[1])/2)# + g.y)/2
			if center is not None:
				g.x = center[0]
				g.y = center[1]



	def clear(self):
		while (len(self.features) > 0 ):
			self.removeFeature(self.features[0])






def trackerIterator(roi, img, similarityThresh, maxIterations, iterations, dissimilarityThresh, similarityBreakThresh, convergeThresh):

	for feature in roi.features:
		
		if feature is None:
			continue

		e = 100000000
		e2= 100000000
		E = None
		mIter = maxIterations
		i = iterations
		

		while(True):

			if mIter < 0 or i < 0:
				print 'stagnation', e, e2
				break

			mIter-=1

			dP, e = feature.trackInverseTranslations(img,similarityBreakThresh, True)
			E2, e2 = feature.errorImage(img, dP)
			feature.e = e2
			dPnorm = np.sqrt(dP[0]*dP[0]+dP[1]*dP[1])
			
			if (e2 > dissimilarityThresh):
				print 'dissimilarityThresh', e2, '>', dissimilarityThresh
				#roi.removeFeature(feature)
				break

			if dPnorm > 100:
				#roi.removeFeature(feature)
				print 'runaway'
				break

			if (e2 < similarityThresh):
				#update template?
				print 'similarity', e, e2
				break

			if dPnorm <= 0.1:
				i-=1
			else:
				i+=1

			if e2 < convergeThresh and dPnorm <= 0.02:
				print 'converged?', dP, e, e2
				break

	return roi

		
def featureAdd(x,y,w=None,h=None):
	global img, parameters, roi

	if w is None:
		w = parameters[5]
	if h is None:
		h = parameters[6]
	feature = Feature(x - int(w/2), y - int(h/2), w, h, img)

	if feature is not None:
		roi.addFeature(feature)



def click(event, x, y, flags, param):
	global roi
	if event == cv2.EVENT_LBUTTONDOWN:
		featureAdd(x,y)





		
#### main ####

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

source = 'camera' # 'video'
record = True


if source == 'camera':
	cap = cv2.VideoCapture('ground.mp4')
else:
	cap = cv2.VideoCapture(0)

roi = RegionOfInterest()


while(True):
	ret, colorImg = cap.read()
	img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
	cv2.imshow('image1',img)
	if record is True:
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		height , width  =  img.shape
		out = cv2.VideoWriter('./output/output' + str(time.time()) +'.avi',fourcc , 25, (width,height))
		cv2.waitKey(0)
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#region = cv2.selectROI(template0)

#roi = [369, 156, 130, 100]
#cv2.destroyWindow('ROI selector')
#window = [[roi[0],roi[1]],[roi[0]+roi[2],roi[1]+roi[3]]]
#windowGlobal = window
#windowSize = [roi[2], roi[3]]

#square features for ground.mp4
#featureAdd(399, 182)
#featureAdd(459, 176)
#featureAdd(410, 182)
#featureAdd(380, 186)
#featureAdd(481, 173)
#featureAdd(459, 152)
#featureAdd(392, 159)
#featureAdd(466, 248)
#featureAdd(411, 257)
#featureAdd(417, 218)
#featureAdd(451, 212)


if True:
	#eyebrows
	featureAdd(394, 165, 40, 5)
	featureAdd(471, 157, 40, 5)
	#nostril
	featureAdd(427, 224, 10, 10)
	#nose center
	featureAdd(432, 197, 5, 50)
	#mouth
	featureAdd(447, 255, 70, 5)
	#jacket
	featureAdd(481, 394, 40, 5)


while(True):
	# Capture frame-by-frame
	ret, colorImg = cap.read()
	img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

	roi = trackerIterator(roi, img, parameters[3], parameters[0], parameters[1], parameters[4], parameters[2], parameters[7])
	roi.normalize()

	showcaseImg = img.copy()

	for feature in roi.features:
		p2 = feature.p2()
		rectangle = [feature.x, feature.y, p2[0], p2[1]]
		cv2.rectangle(showcaseImg,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,255,255) if feature.e < 400 else (0,0,0),1)
		
	cv2.imshow('image1',showcaseImg)
	out.write( cv2.cvtColor(showcaseImg,cv2.COLOR_GRAY2RGB))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if cv2.waitKey(10) & 0xFF == ord('c'):
		roi.clear()



cap.release()
out.release()

np.savetxt('parameters.txt', parameters, fmt='%d')

cv2.destroyAllWindows()


