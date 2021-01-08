import numpy

def getpositions(im):
	leftmost=0
	rightmost=0
	topmost=0
	bottommost=0
	temp=0
	for i in range(224):
		col=im[0:224,i]
		if col.sum()!=0.0:
			rightmost=i
			if temp==0:
				leftmost=i
				temp=1		
	for i in range(224):
		row=im[i,0:224] 
		if row.sum()!=0.0:
			bottommost=i
			if temp==1:
				topmost=i
				temp=2	
	return (leftmost,rightmost,topmost,bottommost)