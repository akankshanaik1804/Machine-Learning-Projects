import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# Read the CSV file
data = pd.read_csv("F:/CMREC/Project/Major Project/poster_data_v3_model.csv")

# Add new columns to the csv to store the new data
data['Identified'] = ''
data['Match'] = ''
data['Accuracy'] = ''

# Define the list of characters
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Define a function to predict the character in a given image
def predict_character(img):
    # Resize and preprocess the image
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = img / 255.0
    img = 1 - img
    img = img.reshape(1, 28, 28, 1)
    # Predict the character using the model
    pred = np.argmax(model.predict(img), axis=-1)
    return characters[pred[0]]

# Load the model
model = load_model("F:/CMREC/Project/Major Project/Major Project Codes/model_bymerge_3convo_v3.h5")

# Set the maximum number of posters to process
max_posters = 30
start_post = 0 #n

# Loop over the poster names and paths
counter = 0
for index in range(start_post, start_post+max_posters):
    try:
        name = data.at[index,'name']
        path = data.at[index,'path']
        # Load the image and preprocess it
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Unable to load image: {}".format(path))
        
        img = cv2.resize(img, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours and sort them from left to right
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # Define the dictionary to store the count of each character
        char_count = {c: 0 for c in characters}

        # Loop over the contours and predict the character in each one
        for ctr in contours:
            # Get the bounding box and ROI
            x, y, w, h = cv2.boundingRect(ctr)
            roi = img[y:y+h, x:x+w]

            # Predict the character and increment the count in the dictionary
            char = predict_character(roi)
            char_count[char] += 1

        # Print the count of each character present in the image
        for char in char_count:
            print(char + ': ' + str(char_count[char] > 0))

        # Get the list of characters to be identified in the current poster
        cposter = list(name)

        # Get the list of characters identified by the model
        fop = [char for char in characters if char_count[char] > 0]

        # Get the list of correctly identified characters
        correct_characters = [char for char in fop if char in cposter]

        # Print the percentage of accuracy
        accuracy = len(correct_characters) / len(cposter) * 100
        print('Accuracy: {}%'.format(accuracy))
        # Print the list of characters to be identified in the current poster
        print('Characters to be identified in the current poster: {}'.format(cposter))

        # Print the list of characters identified by the model
        print('Characters identified by the model: {}'.format(fop))
        data.at[index, 'Identified'] = ' '.join(fop)

        # Print the list of correctly identified characters
        print('Correctly identified characters: {}'.format(correct_characters))
        data.at[index, 'Match'] = ' '.join(correct_characters)

        # Print the percentage of accuracy
        accuracy = len(correct_characters) / len(cposter) * 100
        accuracy_str = str(accuracy)  # Convert accuracy to a string
        print('Accuracy: {}%'.format(accuracy_str))
        data.at[index, 'Accuracy'] = accuracy_str

        # Increment the counter
        counter += 1
        if counter >= start_post + max_posters:
            break
    except Exception as e:
        print("Error processing image {}: {}".format(path, str(e)))

# Save the data into the csv file
data.to_csv("F:/CMREC/Project/Major Project/poster_data_v3_model.csv", index=False)