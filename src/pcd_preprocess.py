import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def remove_roof(pcd):
	# remove roof to show
	show_pcd=deepcopy(pcd)
	p=np.asarray(pcd.points)
	c=np.asarray(pcd.colors)

	# remove points which depth=0
	#print(np.sort(np.unique(p[:,1])))
	mask= (p[:,1]>-0.5)

	p=np.delete(p,mask,axis=0)
	c=np.delete(c,mask,axis=0)
	show_pcd.points = o3d.utility.Vector3dVector(p)
	show_pcd.colors=o3d.utility.Vector3dVector(c)
	return show_pcd,p,c

def remove_ceiling(pcd):
	# remove roof to show
	show_pcd=deepcopy(pcd)
	p=np.asarray(pcd.points)
	c=np.asarray(pcd.colors)

	# remove points which depth=0
	#print(np.sort(np.unique(p[:,1])))
	mask= (p[:,1]<-1.3)

	p=np.delete(p,mask,axis=0)
	c=np.delete(c,mask,axis=0)
	show_pcd.points = o3d.utility.Vector3dVector(p)
	show_pcd.colors=o3d.utility.Vector3dVector(c)
	return show_pcd,p,c

def read_pcd_from_file(pcd_pth,color_type="0255"):
	points=np.load(f"{pcd_pth}/point.npy")
	colors=np.load(f"{pcd_pth}/color{color_type}.npy")

	points=points * 10000/ 255
	# assign to pcd
	pcd=o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors=o3d.utility.Vector3dVector(colors)

	#o3d.visualization.draw_geometries([pcd], window_name='Visualizer Window_downsample', width=resolution, height=resolution)
	
	return pcd

def draw_scatter_graph(points,colors):
	x=points[:,0]
	z=points[:,2]
	print(np.sort(np.unique(x)))
	print(np.sort(np.unique(z)))
	
	plt.scatter(z,x,c=colors,s=2)
	plt.axis('off')
	plt.gca().set_aspect('equal')
	plt.savefig("./map.png")
	plt.show()
	

if __name__=="__main__":
	resolution=512
	color_type="01"
	pcd_pth="./semantic_3d_pointcloud/"
	pcd=read_pcd_from_file(pcd_pth,color_type)
	pcd,_,_=remove_roof(pcd)
	pcd,p,c=remove_ceiling(pcd)

	np.save("./new_points.npy",p)
	np.save(f"./new_colors{color_type}.npy",c)
	
	o3d.visualization.draw_geometries([pcd], window_name='Visualizer Window_downsample', width=resolution, height=resolution)
	
	draw_scatter_graph(p,c)	