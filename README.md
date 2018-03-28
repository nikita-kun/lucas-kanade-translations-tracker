# lucas-kanade-translations-tracker
Lucas-Kanade forward-additive feature tracker for 2D translations.

# Requirements
* opencv

# Usage
To select capturing mode update:
* source = "camera"  #video
* default input video filename is "ground.mp4"

## Interface
* Use trackbars to change parameter values (parameters are saved in parameters.txt)
* Click on the image window to add a new feature
* Press Q to quit
* Press C to remove all features


## Default parameters:
* maxIterations,         Total maximum LK iterations for a feature
* initIterations,        Initial amount of iterations
* similarityBreakThresh, Allowed dissimilarity to skip tracking
* similarityThresh,      Allowed dissimilarity to stop tracking
* dissimilarityThresh,   Dissimilarity threshold to mark feature as an outlier
* featureX,              Default feature width
* featureY,              Default feature height
* convergeThresh,        Allowed dissimilarity for non-settling features
