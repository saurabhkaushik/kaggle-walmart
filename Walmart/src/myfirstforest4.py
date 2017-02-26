'''
"TripType","VisitNumber","Weekday","Upc","ScanCount","DepartmentDescription","FinelineNumber"
'''

import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

train_data=[] # Create a bin to hold our training data.
test_data=[]  # Create a bin to hold our test data.

# Read in CSVs, train and test

with open('../data/train.csv', 'rb') as f1:
    header = f1.next()
    for row in  csv.reader(f1):       # Skip through each row in the csv file
        train_data.append(row)        # Add each row to the data variable
    train_data = np.array(train_data) # Then convert from a list to a NumPy array

with open('../data/test.csv', 'rb') as f2:  # Load in the test csv file
    f2.next()                       # Skip the fist line because it is a header
    for row in csv.reader(f2):      # Skip through each row in the csv file
        test_data.append(row)       # Add each row to the data variable
    test_data = np.array(test_data) # Then convert from a list to an array


dweekdays = list(enumerate(np.unique(train_data[0::, 2])))    
dweekdays_dict = { name : i for i, name in dweekdays }             
train_data[0::, 2] = map( lambda x: dweekdays_dict[x], train_data[0::, 2])

DeptDsrp = list(enumerate(np.unique(train_data[0::, 5])))    
DeptDsrp_dict = { name : i for i, name in DeptDsrp }             
train_data[0::, 5] = map( lambda x: DeptDsrp_dict[x], train_data[0::, 5])

train_data = np.delete(train_data,[1,3,6],1) #remove the name data, cabin and ticket

ids = train_data[0::, 0]
test_data[0::, 4] = map( lambda x: DeptDsrp_dict[x], test_data[0::, 4])

test_data[0::, 1] = map( lambda x: dweekdays_dict[x], test_data[0::, 1])

test_data = np.delete(test_data,[0,2,5],1) # Remove the name data, cabin and ticket

# The data is now ready to go. So lets train then test!

print 'Training '
forest = RandomForestClassifier(n_estimators = 1000)

forest = forest.fit(train_data[0::,1::],\
                    train_data[0::,0])

print 'Predicting'
output = forest.predict(test_data) #predict results using our CLEANED data

predictions_file = open("../output/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["VisitNumber","TripType"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

print "Analysis has Finished"


