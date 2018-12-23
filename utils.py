
import numpy as np
import re


def hot_encode(x):
    new_number = x
    one_hot_number = list()
    digit = [0 for _ in range(0,10)]
    digit[new_number] = 1
    one_hot_number.append(digit)
    return one_hot_number
  
  
def one_hot_encoding_char(str_user):
    complete_characters = "abcdefghijklmnopqrstuvwxyz ,./?!"
    characters_index = dict((c,i) for i,c in enumerate(complete_characters))
    sample_text_index = [characters_index[x] for x in str_user]
    sample_text_onehot_encode = []
    for value in sample_text_index:
        char = np.zeros((len(complete_characters),1))
        char[value] = 1
        sample_text_onehot_encode.append(char)
    return sample_text_onehot_encode
  
  
# Helper functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(S):
    return S * (1 - S)

def clipping_nums(num,thresh):
    if num < -thresh:
        num = -thresh
    elif num > thresh:
        num = thresh
    return num

def decimal_to_binary(number):
    r = 0
    q = 0
    bin_num = []
    if number == 1:
        return 1
    else:
        while q != 1:
            q = number // 2
            r = number % 2
            number = q
            bin_num.append(r)
            if q == 1:
                bin_num.append(q)
        return bin_num[::-1]

def vocab_word(document):
    vocab = set(re.findall(r'\w+',document))
    vocab = dict((w,i) for i,w in enumerate(vocab))
    return vocab

def one_hot_encoding_words(vocab, document):
    document_1hot = np.zeros((1,len(vocab)))
    document_words = re.findall(r'\w+',document)
    for i,word in enumerate(vocab):
        if word in document_words:
            document_1hot[:,i] = 1
    return document_1hot

# Generating random words
def password_generator(length):

    l = length
    vocab = 'abcdefghijklmnopqrstuvwxyz!@0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    chars = list(set(vocab))

    char_idx = {ch:i for i,ch in enumerate(chars)}
    idx_char = { i:ch for i,ch in enumerate(chars)}

    new_idx = np.random.choice(range(0,25),l)
    random_word = ''
    for i in new_idx:
        random_word = random_word + ''.join(idx_char[i])

    return random_word

def y_one_hot(y_labels):
    label_size = 10
    y_data = []
    for i in range(len(y_labels)):
        idx = int(y_labels[i])
        y_temp = np.zeros((label_size))
        y_temp[idx] = 1
        y_data.append(y_temp)
    return y_data

# function to create training batches for X and Y
def batch_data(x_data,y_data,batch_size):
    i = 0
    start = 0
    num_batches = int(x_data.shape[0] / batch_size)
    while i < num_batches:
        x_batch = x_data[start:start+batch_size,:]
        y_batch = y_data[start:start+batch_size,:]
        start += batch_size
        i += 1
        yield x_batch, y_batch
        
# Modifying contents of a file and write it into a new file
import os

old_file_path = "file_path\old"
new_file_path = "file_path\new"
file_list = os.listdir(old_file_path)
for file in file_list:
#     print("Working on:{}".format(file))
    with open(os.path.join(old_file_path,file),'r') as f:
        with open(os.path.join(new_file_path,file),'w') as f1:
          for line in f.readlines():
            if line[0] == '0':
                new_line = '23' + line[1:]
                f1.write(new_line)
            elif line[0] == '1':
                new_line = '24' + line[1:]
                f1.write(new_line)

                
# converting coco anotations json into text
import os
import json
from os import listdir, getcwd
from os.path import join

classes = ["person", "thing",  "backpack", "umbrella", "handbag", "suitcase", "bottle",
            "cup",  "bowl", "chair", "sofa", "pottedplant", "bed", "dining table",
            "toilet", "tvmonitor", "laptop", "remote", "cell phone", "sink", "book",
            "clock", "vase"]

for i, v in enumerate(classes):
    print("{}: {}".format(i, v))

# Creating ratios as required by yolo
def convert_bb(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]*dw
    y = box[1]*dh
    w = box[2]*dw
    h = box[3]*dh
    return (x, y, w, h)

def convert_annotation_to_txt():
    imgs_to_deleted = []
    with open('annotation.json','r') as f:
        data = json.load(f)
        print('hey')
        for item in data['images']:
            image_id = item['id']
            file_name = item['file_name']
            width = item['width']
            height = item['height']
            value = filter(lambda item1: item1['image_id'] == image_id, data['annotations'])

            for item2 in value:
                category_id = item2['category_id']
                value1 = filter(lambda item3: item3['id'] == category_id,data['categories'])
                name = next(value1)['name']
                if name not in classes:
                    imgs_to_deleted.append(file_name)
                else:
                    class_id = classes.index(name)
                    print(class_id)
                    box = item2['bbox']
                    bb = convert_bb((width, height), box)
                    outfile = open('%s.txt'%(file_name[:-4]), 'a+')
                    outfile.write(str(class_id)+" "+" ".join([str(a) for a in bb]) + '\n')
                    outfile.close()

#helper function to extract class labels, annotations, count on each class.		
import os
from collections import Counter
import matplotlib.pyplot as plt
import csv

classes = {0:"person", 1:"thing",  2:"backpack", 3:"umbrella", 4:"handbag", 5:"suitcase", 6:"bottle",7:"cup",  8:"bowl", 9:"chair", 10:"sofa", 11:"pottedplant", 12:"bed", 13:"diningtable",
           14:"toilet", 15:"tvmonitor", 16:"laptop", 17:"remote", 18:"cellphone", 19:"sink", 20:"book",21:"clock", 22:"vase"}


file_path = "/home/user/Development/yolo/darknet/data/coco/labels/train"
files = os.listdir(file_path)

class_list = []
max_persons = []
file_persons = []

for file_name in files:
	person_counter = 0
	with open(os.path.join(file_path,file_name), 'r') as f:
	
		for line in f.readlines():
			line = line.split(' ')
			temp = line[0]
			class_list.append(classes[int(temp)])
			if int(temp) in [0, 23, 24]:
				person_counter += 1
		if person_counter > 2:
			max_persons.append(person_counter)
			file_persons.append(file_name)
			
l1 = Counter(class_list).keys()
l2 = Counter(class_list).values()
data = dict(zip(l1, l2))

with open('train.csv', 'w') as train_csv:
	writer = csv.writer(train_csv)
	for key, value in data.items():
		writer.writerow([key, value])

max_persons_data = dict(zip(file_persons, max_persons))
with open('max_persons_train.csv', 'w') as max_train:
	writer = csv.writer(max_train)
	for key, value in max_persons_data.items():
		writer.writerow([key, value])

l3 = Counter(max_persons).keys()
l4 = Counter(max_persons).values()
max_p = dict(zip(l3, l4))

with open('max_people_train.csv', 'w') as maxp_train:
	writer = csv.writer(maxp_train)
	for key, value in max_p.items():
		writer.writerow([key, value])

plt.bar(data.keys(), data.values())
plt.savefig('train_hist.png', dpi=350)
plt.show()

plt.plot(max_p.keys(), max_p.values(), 'b.')
plt.show()

#
#
#


# Python wrapper to run darknet object detection on multiple images.

import darknet
import cv2
import os

# config and image paths
CFG = "/your/path/darknet/cfg/yolov2.cfg"
WEIGHTS = "/your/path/darknet/yolov2_org_weights/yolov2.weights"
META = "/your/path/darknet/cfg/coco.data"
image_path = "/your/path/darknet/data/images"
target_path = "/your/path/darknet/data/new_images"

darknet.set_gpu(0)
net = darknet.load_net(str.encode(CFG), str.encode(WEIGHTS), 0)
meta = darknet.load_meta(str.encode(META))
class_label = 'person'
images = os.listdir(image_path)
bb = []
img_id = 1
with open('labels_data_dist.txt', 'w') as f:

    for img in images:
        print("image number: ", img_id)
        img_path = os.path.join(image_path, img)
        image = cv2.imread(img_path)
        #cv2.imshow("image", image)
        d_imgpath = os.path.join(str.encode(image_path), str.encode(img))
        res = darknet.detect(net, meta, d_imgpath)
        for r in res:
            if r:
                name = r[0].decode('utf-8')
                if name == class_label:
                    bb = r[2]
                    print("class label: ", name)
                    print("bounding box: ", bb)
                    x = r[2][0]
                    y = r[2][1]
                    w = r[2][2]
                    h = r[2][3]
                    x_max = int(round((2*x+w)/2))
                    x_min = int(round((2*x-w)/2))
                    y_max = int(round((2*y-h)/2))
                    y_min = int(round((2*y+h)/2))
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    #cv2.imshow("image", image)
                    cv2.imwrite(os.path.join(target_path, 'image' + str(img_id) + '.jpg'), image)
                    f.write('image' + str(img_id) + '\n')
        img_id += 1
        #cv2.waitKey(0)
cv2.destroyAllWindows()

import os
from collections import Counter
import matplotlib.pyplot as plt

images = []

with open('labels_data_dist.txt', 'r+') as f:
    images = [line[:-1] for line in f.readlines()]

data = Counter(images)
new_data = Counter(data.values())

#plotting graph
plt.xlabel("Persons per image")
plt.ylabel("Number of images")
plt.bar(new_data.keys(), new_data.values())
plt.savefig('person_per_image.png', dpi=300)
plt.show()


#
#
#




