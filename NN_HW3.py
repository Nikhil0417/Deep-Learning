import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

def relu(z):
	return z*(z>0)

x = np.array([[-1, -1, -1]
	,[-1, -1, 1]
	,[-1, 1, -1]
	,[-1, 1, 1]
	,[1, -1, -1]
	,[1, -1, 1]
	,[1, 1, -1]
	,[1, 1, 1]])
X = x.T
Y = np.array([[0, 1, 1, 0, 1, 0, 0, 1]])
m = Y.size
print(m)
print(X.shape)
print(Y.shape)
L = []

l = 0.2
#layer-1
w1 = np.random.random((4,3)) - .5
b1 = np.zeros((4,1))
print("layer1 weight",w1)

#layer-2
w2 = np.random.random((1,4)) - .5
b2 = 0
#J = 0.1
#for loop starts here
for i in range(1,40):
#while J > 0.01:
	snc = np.dot(w1,X)
	z1 = snc + b1

	a1 = relu(z1)

	z2 = np.dot(w2,a1) + b2

	a2 = sigmoid(z2)
	print("a2 shape",a2.shape)

	J = -(np.sum(Y*(np.log(a2))+((1-Y)*np.log(1-a2))))/m
	#print("cost",J)
	L.append(J)
	print(J)
	dz2 = (a2-Y)

	dw2 = np.dot(dz2,a1.T)

	db2 = dz2

	w2 = w2 - l*dw2
	b2 = b2 - l*db2

	da = np.dot(w2.T,dz2)

	bp_sig = relu(dz2)
	dz1 = np.multiply(da,bp_sig)
	#dz1 = relu(a1)

	dw1 = np.dot(dz1, X.T)
	db1 = dz1

	w1 = w1 - l*dw1
	b1 = b1 - l*db1
print(L)
print("y hat",a2)
print("Y",Y)
plt.plot(L)
plt.show()