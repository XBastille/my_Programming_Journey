import numpy as np
import cv2
from functions import*
def projection_mat(K,H):
	h1=H[:,0]
	h2=H[:,1]

	K=np.transpose(K)

	K_inv=np.linalg.inv(K)
	a=np.dot(K_inv,h1)
	c=np.dot(K_inv,h2)
	lamda=1/((np.linalg.norm(a)+np.linalg.norm(c))/2)

	Bhat=np.dot(K_inv,H)

	B = 1*Bhat if np.linalg.det(Bhat)>0 else -1*Bhat
	b1=B[:,0]
	b2=B[:,1]
	b3=B[:,2]
	r1=lamda*b1
	r2=lamda*b2
	r3=np.cross(r1,r2)
	t=lamda*b3

	return np.dot(K,(np.stack((r1,r2,r3,t), axis=1)))



def cubePoints(corners, H, P, dim):
	new_points=[]
	x=[]
	y=[]
	for point in corners:
		x.append(point[0])
		y.append(point[1])
	H_c = np.stack((np.array(x),np.array(y),np.ones(len(x))))

	sH_w=np.dot(H,H_c)

	H_w=sH_w/sH_w[2]

	P_w=np.stack((H_w[0],H_w[1],np.full(4,-dim),np.ones(4)),axis=0)

	sP_c=np.dot(P,P_w)
	P_c=sP_c/(sP_c[2])

	return [[int(P_c[0][i]),int(P_c[1][i])] for i in range(4)]


def drawCube(tagcorners, new_corners,frame,face_color,edge_color,flag):
	thickness=5
	if not flag:
		contours = makeContours(tagcorners,new_corners)
		for contour in contours:
			cv2.drawContours(frame,[contour],-1,face_color,thickness=-1)

	for i, point in enumerate(tagcorners):
		cv2.line(frame, tuple(point), tuple(new_corners[i]), edge_color, thickness) 

	for i in range (4):
		if i==3:
			cv2.line(frame,tuple(tagcorners[i]),tuple(tagcorners[0]),edge_color,thickness)
			cv2.line(frame,tuple(new_corners[i]),tuple(new_corners[0]),edge_color,thickness)
		else:
			cv2.line(frame,tuple(tagcorners[i]),tuple(tagcorners[i+1]),edge_color,thickness)
			cv2.line(frame,tuple(new_corners[i]),tuple(new_corners[i+1]),edge_color,thickness)

	return frame


def makeContours(corners1,corners2):
	contours = []
	for i in range(len(corners1)):
		p1=corners1[i]
		if i==3:
			p2 = corners1[0]
			p3 = corners2[0]
		else:
			p2 = corners1[i+1]
			p3 = corners2[i+1]
		p4 = corners2[i]
		contours.append(np.array([p1,p2,p3,p4], dtype=np.int32))
	contours.extend(
		(
			np.array(
				[corners1[0], corners1[1], corners1[2], corners1[3]],
				dtype=np.int32,
			),
			np.array(
				[corners2[0], corners2[1], corners2[2], corners2[3]],
				dtype=np.int32,
			),
		)
	)
	return contours


def getCorners(frame):
	[all_cnts,cnts] = findcontours(frame,180)
	[tag_cnts,corners] = approx_quad(cnts)

	tag_corners={}
	dim = 200
	for i,tag in enumerate(corners):
		H = homography(tag,dim)
		H_inv = np.linalg.inv(H)

		#get squared tile
		square_img = fastwarp(H_inv,frame,dim,dim)
		imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
		ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)
		[tag_img,id_str,orientation] = encode_tag(square_img)

		ordered_corners=[]

		if orientation==0:
		    ordered_corners=tag

		elif orientation==1:
		    ordered_corners.append(tag[1])
		    ordered_corners.append(tag[2])
		    ordered_corners.append(tag[3])
		    ordered_corners.append(tag[0])

		elif orientation==2:
		    ordered_corners.append(tag[2])
		    ordered_corners.append(tag[3])
		    ordered_corners.append(tag[0])
		    ordered_corners.append(tag[1])

		elif orientation==3:
		    ordered_corners.append(tag[3])
		    ordered_corners.append(tag[0])
		    ordered_corners.append(tag[1])
		    ordered_corners.append(tag[2])

		tag_corners[id_str] = ordered_corners

	return tag_corners


def getTopCorners(bot_corners):
	K = np.array([[1406.08415449821,0,0],
			[2.20679787308599, 1417.99930662800,0],
 			[1014.13643417416, 566.347754321696,1]])

	top_corners = {}

	for tag_id, corners in bot_corners.items():
		H = homography(corners,200)
		H_inv = np.linalg.inv(H)
		P = projection_mat(K,H_inv)
		top_corners[tag_id] = cubePoints(corners, H, P, 200)

	return top_corners


def avgCorners(past, current, future):
	diff = 50
	average_corners={}
	for tag in current:
		templist=[current[tag]]
		if past == []:
			pass
		elif tag in past[-1]:
			templist.extend(d[tag] for d in past if tag in d)
		if tag in future[0]:
			templist.extend(d[tag] for d in future if tag in d)
		newcorners=[]
		c1x=c1y=c2x=c2y=c3x=c3y=c4x=c4y=0

		for allcorners in templist:
		    c1x+=allcorners[0][0]
		    c1y+=allcorners[0][1]
		    c2x+=allcorners[1][0]
		    c2y+=allcorners[1][1]
		    c3x+=allcorners[2][0]
		    c3y+=allcorners[2][1]
		    c4x+=allcorners[3][0]
		    c4y+=allcorners[3][1]

		newcorners=np.array([[c1x,c1y],[c2x,c2y],[c3x,c3y],[c4x,c4y]])
		newcorners=np.divide(newcorners,len(templist))
		newcorners=newcorners.astype(int)
		newcorners=np.ndarray.tolist(newcorners)
		teleport = False
		for i in range(4):
		    orig_x = current[tag][i][0]
		    orig_y = current[tag][i][1]
		    new_x = newcorners[i][0]
		    new_y = newcorners[i][1]
		    if (abs(orig_x-new_x) > diff) or (abs(orig_y-new_y) > diff):
		        teleport = True
		average_corners[tag] = current[tag] if teleport else newcorners
	return average_corners
