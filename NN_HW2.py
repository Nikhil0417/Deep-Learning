import numpy as np
import matplotlib.pyplot as plt

def plotter():
	plt.scatter(x[:,0],x[:,1],marker='.')
	plt.scatter(w[:,0],w[:,1],color='r',marker='+')
	plt.plot(w[:,0],w[:,1],color = "yellow")
	plt.title('Kohonen self-organizing map')
	plt.xlabel('W1')
	plt.ylabel('W2')
	plt.xlim(xmin=-1, xmax=1)
	plt.ylim(ymin=-1, ymax=1)
l = 0.5	#initial learning rate
p = []	#empty array
s = []
while len(p) != 100:
	x1 = np.random.random() - .5
	x2 = np.random.random() - .5
	#x1 = np.random.seed(-0.5,0.5)
	#x2 = np.random.seed(-0.5,0.5)
	#q = np.random.uniform(-1,1)
	#w.append([q, q])
	#x = np.array([x1, x2])
	z = x1**2 + x2**2
	if z < 0.25:
		p.append([x1, x2])

while len(s) != 50:
	q1 = np.random.uniform(-1,1)
	#print(q1)
	q2 = np.random.uniform(-1,1)
	s.append([q1, q2])

#print(p)
x = np.array(p)
#print(x.size)
print(x)
w = np.array(s)
print(w.size)
#w = np.transpose(d)
#print(d)
print("")
print(w)

plt.scatter(x[:,0],x[:,1],marker='.')
plt.scatter(w[:,0],w[:,1],color='r',marker='+')
plt.plot(w[:,0],w[:,1],color = "yellow")
plt.title('Kohonen self-organizing map')
plt.xlabel('W1')
plt.ylabel('W2')
plt.xlim(xmin=-1, xmax=1)
plt.ylim(ymin=-1, ymax=1)
plt.figure()

for t in range(0,100):
	for i in range(0,100):
		e = []
		for j in range(0,50):
			v = 0
			for k in range(0,2):
				#print("input",p[i][k])
				#print("weight",w[j][k])
				u = (w[j][k] - p[i][k])**2
				v = v + u
				#print("dist",u)
				#print(i, j, k)
			e.append(v)
			f = np.array(e)
			J = np.argmin(f)
			#print("dist array",f)
			#print("min cluster",J)
		for m in range(0,2):
			#print("Hey",J)
			w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
			if J-1 == -1 or J-2 == -1: #or J-3 == -1:
				for a in range(1,3):
					w[J+a][m] = w[J+a][m] + l*(p[i][m] - w[J+a][m])
				# w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
				# w[J+2][m] = w[J+2][m] + l*(p[i][m] - w[J+2][m])
				# w[J+1][m] = w[J+1][m] + l*(p[i][m] - w[J+1][m])
			elif J+1 == 50 or J+2 == 50: #or J+3 == 50:
				for a in range(1,3):
					w[J-a][m] = w[J-a][m] + l*(p[i][m] - w[J-a][m])
				# w[J-2][m] = w[J-2][m] + l*(p[i][m] - w[J-2][m])
				# w[J-1][m] = w[J-1][m] + l*(p[i][m] - w[J-1][m])
				# w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
			else:
				for a in range(1,3):
					w[J-a][m] = w[J-a][m] + l*(p[i][m] - w[J-a][m])
				#w[J-1][m] = w[J-1][m] + l*(p[i][m] - w[J-1][m])
				#w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
					w[J+a][m] = w[J+a][m] + l*(p[i][m] - w[J+a][m])
				#w[J+2][m] = w[J+2][m] + l*(p[i][m] - w[J+2][m])
		#J=0
	if t == 9 or t == 19 or t == 29:
		plt.scatter(x[:,0],x[:,1],marker='.')
		plt.scatter(w[:,0],w[:,1],color='r',marker='+')
		plt.plot(w[:,0],w[:,1],color = "yellow")
		plt.title('Kohonen self-organizing map')
		plt.xlabel('W1')
		plt.ylabel('W2')
		plt.xlim(xmin=-1, xmax=1)
		plt.ylim(ymin=-1, ymax=1)
		plt.figure()
	#print("Weights Updated")
	l = l - 0.0049

	
print("")
print(w)

plt.scatter(x[:,0],x[:,1],marker='.')
plt.scatter(w[:,0],w[:,1],color='r',marker='+')
plt.plot(w[:,0],w[:,1],color = "yellow")
plt.title('Kohonen self-organizing map')
plt.xlabel('W1')
plt.ylabel('W2')
plt.xlim(xmin=-1, xmax=1)
plt.ylim(ymin=-1, ymax=1)
plt.show()
# for t in range(0,100):
# 	for i in range(0,100):
# 		e = []
# 		for j in range(0,50):
# 			v = 0
# 			for k in range(0,2):
# 				#print("input",p[i][k])
# 				#print("weight",w[j][k])
# 				u = (w[j][k] - p[i][k])**2
# 				v = v + u
# 				#print("dist",u)
# 				#print(i, j, k)
# 			e.append(v)
# 			f = np.array(e)
# 			J = np.argmin(f)
# 		#print("dist array",f)
# 		#print("min cluster",J)
# 		for m in range(0,2):
# 			if J-1 == -1:
# 				w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
# 				w[J+1][m] = w[J+1][m] + l*(p[i][m] - w[J+1][m])
# 				#w[J+3][m] = w[J+3][m] + l*(p[i][m] - w[J+3][m])
# 			elif J+1 == 50:
# 				w[J-1][m] = w[J-1][m] + l*(p[i][m] - w[J-1][m])
# 				w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
# 			else:
# 				w[J-1][m] = w[J-1][m] + l*(p[i][m] - w[J-1][m])
# 				w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
# 				w[J+1][m] = w[J+1][m] + l*(p[i][m] - w[J+1][m])
# 		#J=0
		
# 	#print("Weights Updated")
# 	l = l - 0.0049

	
# print("")
# print(w)

# for t in range(0,100):
# 	for i in range(0,100):
# 		e = []
# 		for j in range(0,50):
# 			v = 0
# 			for k in range(0,2):
# 				u = (w[j][k] - p[i][k])**2
# 				v = v + u
# 			e.append(v)
# 			f = np.array(e)
# 			J = np.argmin(f)
# 		for m in range(0,2):
# 			w[J][m] = w[J][m] + l*(p[i][m] - w[J][m])
# 		#J=0
# 		# if t%10 == 0:
# 		# 	plt.subplot(221)
# 		# 	plt.scatter(x[:,0],x[:,1])
# 		# 	plt.plot(w[:,0],w[:,1],color = "red")
# 		# 	plt.title("Epoch")
# 			#plt.figure()
# 	#print("Weights Updated")
# 	l = l - 0.0049

	
# print("")
# print(w)


# plt.scatter(x[:,0],x[:,1],marker='.')
# plt.scatter(w[:,0],w[:,1],color='r',marker='+')
# plt.plot(w[:,0],w[:,1],color = "yellow")
# plt.title('Kohonen self-organizing map')
# plt.xlabel('W1')
# plt.ylabel('W2')
# plt.xlim(xmin=-1, xmax=1)
# plt.ylim(ymin=-1, ymax=1)
# plt.show()