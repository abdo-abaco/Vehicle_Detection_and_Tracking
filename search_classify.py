import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_function import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

    
# Reading the car and non-car images
cars = []
notcars = []

notcars = notcars + glob.glob('non-vehicles/GTI/*.png') + glob.glob('non-vehicles/Extras/*.png') 
cars =  cars + glob.glob('vehicles/GTI_Right/*.png') + glob.glob('vehicles/GTI_Far/*.png') + glob.glob('vehicles/KITTI_extracted/*.png')
cars = cars + glob.glob('vehicles/GTI_MiddleClose/*.png') + glob.glob('vehicles/GTI_Left/*.png')


# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
print(len(cars))
print(len(notcars))


### TODO: Tweak these parameters and see how the results change.
# Define feature parameters

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True



test_cars = cars
test_notcars = notcars

car_features =   extract_features(test_cars, 
                                  color_space=color_space,
                                  spatial_size=spatial_size,
                                  hist_bins=hist_bins,
                                  orient=orient,
                                  pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel,
                                  spatial_feat=spatial_feat,
                                  hist_feat=hist_feat,
                                  hog_feat=hog_feat,
                                  vis=False)

notcar_features =  extract_features(test_notcars, 
                                    color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,                                                        
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat,
                                    vis=False)





# Here the training begins.
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler to normalize the data
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 5))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


dist_pickle = {}
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["scaler"] = X_scaler
dist_pickle["svc"] = svc
pickle.dump( dist_pickle, open( "svc_pickle.p", "wb" ) )


plt.show()



