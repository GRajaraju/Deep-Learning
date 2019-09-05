import os
from collections import Counter
import matplotlib.pyplot as plt
import csv

classes = {0:"person", 1:"thing",  2:"backpack", 3:"umbrella", 4:"handbag", 5:"suitcase", 6:"bottle",7:"cup",  8:"bowl", 9:"chair", 10:"sofa", 11:"pottedplant", 12:"bed", 13:"diningtable",
           14:"toilet", 15:"tvmonitor", 16:"laptop", 17:"remote", 18:"cellphone", 19:"sink", 20:"book",21:"clock", 22:"vase", 23:"patient", 24:"provider"}


file_path = "file/path"
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
