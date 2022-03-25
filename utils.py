import pandas as pd
import cv2

def compute_precision(csv_path, threshold, metric):
    df = pd.read_csv(csv_path, delimiter=";")
    df = df.reset_index()  # make sure indexes pair with number of rows
    good_predictions = 0
    for index, row in df.iterrows():
        test_image_path = row['test_image_path']
        train_image_path = row['train_image_path']
        is_similar = bool(row['label'])
        is_possibly_similar = find_similarity(train_image_path, test_image_path, threshold, metric)
        if (is_similar == is_possibly_similar):
            good_predictions += 1
    
    return good_predictions / df.shape[0]

def find_similarity(train_image_path, test_image_path, threshold, metric):
    image_train = cv2.imread(train_image_path)
    image_test = cv2.imread(test_image_path)

    # Convert the training image to RGB
    training_gray = cv2.cvtColor(image_train, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    # Perform the matching between the ORB descriptors of the training image and the test image
    matches = bf.knnMatch(train_descriptor,test_descriptor,k=2)
    good_matches = find_good_matches(matches)
    similarity = metric(good_matches, matches, train_descriptor, test_descriptor)

    return similarity > threshold
    

def find_good_matches(matches):
    good = []
    for (m1, m2) in matches: # for every descriptor, take closest two matches
        if m1.distance < 0.7 * m2.distance: # best match has to be this much closer than second best
            good.append(m1)
    return good


