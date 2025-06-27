from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris= datasets.load_iris()

#printing the details about the dataset
'''print(iris.keys())
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

print(iris.target)
print(iris.data)'''

features= iris.data
labels= iris.target

#printing description of dataset
'''print(features, labels)
print(features[1], labels[1])
print(iris.DESCR)'''

#training the classifier
clf= KNeighborsClassifier()
clf.fit(features, labels)

#getting data for new dataset
print("\n\n")
a=float(input('Enter SEPAL LENGTH: '))
b=float(input('Enter SEPAL WIDTH: '))
c=float(input('Enter PETAL LENGTH: '))
d=float(input('Enter PETAL WIDTH: '))

print("\n\nFollowing is how iris types are marked:\n\nIris Setosa- [0]\nIris Versicolour- [1]\nIris Virginia- [2]\n\n")

newdata=[a,b,c,d]

#getting classifications for new datasets
pred=clf.predict([newdata])
print('RESULT: ',pred) 

#for accuracy but have to create new dataset with actual labels
'''accuracy= accuracy_score(labels, pred)
print("model accuracy: ", accuracy)'''
