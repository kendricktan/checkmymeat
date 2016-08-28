from skimage import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.svm import SVC

import cv2, glob
import numpy as np

# Get normalized histogram
def describe(image, mask = None):
    hist = cv2.calcHist([image], [0, 1, 2], mask, [8,8,8], [0, 256, 0, 256, 0, 256])    
    cv2.normalize(hist, hist)    
    return hist.flatten()

data = []
target = []

# Sort through images
image_paths = sorted(glob.glob("meat/*.jpg"))
for image_path in image_paths:    
    # Read image
    image = io.imread(image_path)    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)    

    # Use otsu to mask
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)

    features = describe(image, mask=mask)    

    # Append to dataset
    data.append(features)
    target.append(image_path.split("_")[-2])


# Get target names
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

model = RandomForestClassifier(n_estimators = 33, criterion='entropy')

trainData, testData, trainTarget, testTarget = cross_validation.train_test_split(data, target, test_size=0.4)
model.fit(trainData, trainTarget)
#print(classification_report(testTarget, model.predict(testData), target_names = targetNames))

def predict(url):
    global model      
    # Read image
    image = io.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)    

    # Use otsu to mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)

    # Get features
    features = describe(image, mask)

    # Predict it
    result = model.predict([features])
    probability = model.predict_proba([features])[0][result][0]    
    print(model.predict_proba([features])[0][result][0])
    state = le.inverse_transform(result)[0]
    return {'type': state, 'confidence': probability}    