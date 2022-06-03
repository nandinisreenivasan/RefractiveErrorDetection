import csv
from collections import Counter
filename='myfile.txt'
lst=[]

with open(filename, 'r') as f:
  column = (row[0] for row in csv.reader(f))
  print("Most frequent value: {0}".format(Counter(column).most_common()[0][0]))
  
lst.append(format(Counter(column).most_common()[0][0]))
