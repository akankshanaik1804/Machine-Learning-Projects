import cv2
import numpy as np
from keras.models import load_model
model = load_model("F:/CMREC/Project/3rd Year/Semester 2/Mini Project/EMNIST/my_model_bymerge.h5")
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
results = {}
img = cv2.imread("F:/CMREC/Project/3rd Year/Semester 2/Mini Project/EMNIST/Posters/Air.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10: 
        character = gray[y:y+h, x:x+w]
        character = cv2.resize(character, (28, 28))
        character = np.reshape(character, (1, 28, 28, 1))
        character = character / 255.0
        prediction = model.predict(character)
        predicted_label = np.argmax(prediction)
        predicted_character = characters[predicted_label]
        if predicted_character in results:
            results[predicted_character] += 1
        else:
            results[predicted_character] = 1
for character in characters:
    if character in results:
        print(character, end=", ")
