from sklearn import tree

cls = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[198, 80, 10], [178, 79, 8], [106, 60, 7], [145, 52, 10], [160, 95, 8],
     [190, 10, 8], [172, 60, 10], 
     [117, 20, 3], [195, 50, 5], [171, 71, 4], [184, 25, 10],[199,90,9]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male','male']

#training our data
cls = cls.fit(X, Y)
#making predictions
prediction = cls.predict([[160, 50, 30]])
print(prediction)

