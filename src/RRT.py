import argparse
import random

import numpy as np
import matplotlib.pyplot as plt



from cv2 import imread,imwrite
import networkx as nx
from networkx.classes import graph
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

def RRT(p_start,target_label,np_map,theta=0.3,step_size=25,thres_d=30):

	def get_random(theta):
		# random select a point as target
		if random.random()<theta:
			p_rand=p_target
		else:
			p_rand=[int(random.random()*480), int(random.random()*640)]
		return p_rand

	

	def get_nearest(tree,p_rand):
		# find a existed node which is the nearest one close to the target 
		tree_nodes=np.array(tree.nodes())
		n = NearestNeighbors(n_neighbors=1,metric="euclidean")
		n.fit(tree)
		dis, ind = n.kneighbors([p_rand], return_distance=True)
		p_near=tree_nodes[ind.ravel()]
		return p_near.ravel(),dis.ravel()

	def get_new_node(p_near,p_rand,dis,step_size=step_size):
		# get p_new
		d=(p_rand-p_near)/dis*step_size
			
		p_new=p_near+d # one step along the direction
		p_new=p_new.ravel()

		p_new=[int(p_new[0]),int(p_new[1])]
		# this offset to make sure the distance<=step_size
		
		if d[0]<0:
			p_new[0]+=1
		if d[1]<0:
			p_new[1]+=1
		
		return p_new

	
	def check_achievable(p_near,p_new,np_map,n_sample=25):
		# check if it is achievable
		if p_new[0]>=w or p_new[0]<0 or p_new[1]>=h or p_new[1]<0:
			# out of map
			return False

		d=p_new-p_near
		d_step=d.ravel()/n_sample

		for i in range(n_sample):
			p=p_near+d_step*i
			p=[int(p[0]),int(p[1])]
			
			if (np_map[p[1]][p[0]]!=[1,1,1,1]).any():
				# is wall or object
				return False
		return True


	def check_mission(tree,p_target,thres_d=10):
		n = NearestNeighbors(n_neighbors=1,metric="euclidean")
		n.fit(tree)
		dis, ind = n.kneighbors([p_target], return_distance=True)
		if dis.ravel()<thres_d:
			p=np.array(tree.nodes())[ind.ravel()]
			return True,p.ravel()

		return False,None

	
	# map range
	h,w=np_map.shape[0],np_map.shape[1]

	# show start ans goal
	plt.imshow(np_map)
	plt.scatter(p_start[0],p_start[1], marker="v", color="green")
	plt.scatter(p_target[0],p_target[1], marker="*", color="black")
	plt.axis('off')
	plt.show()

	# create Tree
	G = nx.Graph()
	G.add_node(tuple(p_start))	

	
	# start RRT
	count=0
	while tuple(p_target) not in G:
		(flag,p)= check_mission(G,p_target,thres_d)
		if flag:
			G.add_node(tuple(p_target))
			G.add_edge(tuple(p), tuple(p_target))
			break

		if count>=1500:
			# reset
			count=0
			G = nx.Graph()
			G.add_node(tuple(p_start))

		# start search
		p_rand=get_random(theta)
		p_near,dis=get_nearest(G,p_rand)
		p_new=get_new_node(p_near,p_rand,dis,step_size)

		if check_achievable(p_near,p_new,np_map,step_size):
			G.add_node(tuple(p_new))
			G.add_edge(tuple(p_near), tuple(p_new))

		count+=1
		if count%100==0:
			print(f"iter: {count}")
			draw_path(G,None,np_map)



	draw_path(G,None,np_map)
	return G


def get_taget_position(target_label):
	target_color={
		"refrigerator": (255, 0, 0), 
		"rack": (0, 255, 133), 
		"cushion": (255, 9, 92), 
		"lamp": (160, 150, 20),
		"cooktop": (7, 255, 224)
		}

	target=[0,0]
	if target_label=="refrigerator":
		target=[253,249]
	elif target_label=="cushion":
		target=[505,253] 
	else:
		c=target_color[target_label]
		c=(c[0]/255,c[1]/255,c[2]/255)

		mask= (np_map[:,:,0]==c[0]) & (np_map[:,:,1]==c[1]) & (np_map[:,:,2]==c[2])
		out=np.where(mask==True)
		x=int(np.mean(out[1]))
		y=int(np.mean(out[0]))
		target=[x,y]

	print(f"search for {target_label}, set target at {target}")
	return target

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    return (ix, iy)

def draw_path(tree,simple_pth,np_map):
	plt.imshow(np_map)
	
	
	for e in tree.edges():
		#draw edges
		pa,pb=list(e[0]),list(e[1])
		plt.plot([pa[0],pb[0]],[pa[1],pb[1]],color="black")

	for n in tree.nodes():
		#draw nodes
		plt.scatter(n[0],n[1], marker="o", color="hotpink",s=10)

	if simple_pth:
		for i in range(len(simple_pth)-1):
			pa,pb=simple_pth[i],simple_pth[i+1]
			plt.plot([pa[0],pb[0]],[pa[1],pb[1]],color="red")

		
	plt.scatter(p_start[0],p_start[1], marker="v", color="green")
	plt.scatter(p_target[0],p_target[1], marker="*", color="green")

	plt.axis('off')
	plt.savefig(f'./{args.target}.png')
	plt.show()


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--map_pth", default="./map.png")
	parser.add_argument("-t","--target", default="cushion",type=str, 
		choices={"refrigerator", "rack", "cushion", "lamp","cooktop"},
		help="target label to achieve")
	args = parser.parse_args()

	np_map=plt.imread(args.map_pth)

	# get target position
	p_target=get_taget_position(args.target)


	# plot map to choose start point
	fig, ax = plt.subplots()	
	plt.imshow(np_map)
	plt.scatter(p_target[0],p_target[1], marker="*", color="green")
	fig.canvas.mpl_connect('button_press_event', onclick)
	plt.axis('off')
	plt.show()

	# get start point
	global ix,iy
	p_start=[int(ix),int(iy)]
	print(f"start from {p_start}")
	
	

	# start searching
	search_G=RRT(p_start,p_target,np_map)
	answer=nx.all_simple_paths(search_G,tuple(p_start),tuple(p_target))
	answer=list(answer)[0]
	draw_path(search_G,answer,np_map)

	# save as txt
	with open(f"./{args.target}_Path.txt","w+") as f:
		for p in answer:
			f.write(f"{str(p)}\n")



	