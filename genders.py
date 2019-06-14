from sklearn import tree

cls = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 10], [177, 70, 8], [160, 60, 7], [154, 54, 8], [166, 65, 9],
     [190, 90, 10], [175, 64, 9], 
     [177, 70, 8], [159, 55, 7], [171, 75, 8], [181, 85, 10],[195,50,7]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male','male']

#training our data
cls = cls.fit(X, Y)
#making predictions
prediction = cls.predict([[160, 50, 30]])
print(prediction)

