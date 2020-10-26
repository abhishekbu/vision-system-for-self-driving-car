import os
import tqdm
from random import sample

img_dir = "./dataset/img/"
xml_dir = "./dataset/xml/"

# Percentage of images to be used for the test set
percentage_test = 25

# Create and/or truncate train.txt and test.txt
file_train = open('train_tl.txt', 'w')
file_test = open('test_tl.txt', 'w')

direc = os.listdir(img_dir)
direc = sample(direc, len(direc))

counter = 1
index_test = round(100 / percentage_test)

for f in tqdm.tqdm(direc):
    if f.endswith(".jpg"):
        name = f.rstrip(".jpg")
        if os.path.exists(xml_dir+name+".xml"):
            if counter == index_test:
                counter = 1
                file_test.write(name + "\n")
            else:
                file_train.write(name + "\n")
                counter = counter + 1
