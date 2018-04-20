import numpy as np

s1 = np.array([1, 4, 16, 8, 2, 4]) #[2,4]
s2 = np.array([1, 9, 16, 12, 3, 4]) #[3,4]
s3 = np.array([1, 144, 16, 48, 12, 4]) #[12,4]
s4 = np.array([1, 169, 16, 52, 13, 4]) #[13,4]

s5 = np.array([1, 36, 16, 24, 6, 4]) #[6,4]
s6 = np.array([1, 49, 16, 28, 7, 4]) #[7,4]

w = np.array([0,0,0,0,0,0])	#step-0
wT = np.transpose(w)
l = 0.7

for p in range(1, 30):
	count = 0
	print("Iteration",p)
#step-1
	x = s1
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w
		count += 1
		print(count)
	elif (i<=0):
		w = w + l*2*x
	print(w)

#step-2
	x = s2
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w
		count += 1
		print(count)
	elif (i<=0):
		w = w + l*2*x
	print(w)

#step-3
	x = s3
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w
		count += 1
		print(count)
	elif (i<=0):
		w = w + l*2*x
	print(w)

#step-4
	x = s4
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w
		count += 1
		print(count)
	elif (i<=0):
		w = w + l*2*x
	print(w)

#step-5
	x = s5
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w - l*2*x
	elif (i<=0):
		w = w
		count += 1
		print(count)
	print(w)

#step-6
	x = s6
	i = w.dot(x)
	print("J=",i)
	if (i>0):
		w = w - l*2*x
	elif (i<=0):
		w = w
		count += 1
		print(count)
	print(w)

	if count == 6:
		break