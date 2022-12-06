import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json

import argparse
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument("--map_pth", default="./map.png")
parser.add_argument("-t","--target", default="cushion",type=str, 
    choices={"refrigerator", "rack", "cushion", "lamp","cooktop"},
    help="target label to achieve")
parser.add_argument("-d","--data_path", default=".",type=str, help="path and map location")
args = parser.parse_args()
print(args.target)



# read map
map_with_bg=np.zeros((512,640,3))
np_map=cv2.imread(f"{args.data_path}/{args.target}.png")
(x,y,_)=np_map.shape
map_with_bg[:x,:y]=np_map
map_with_bg=map_with_bg[:,64:-64,:]




# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "Replica-Dataset/replica_v1/apartment_0/habitat/mesh_semantic.ply"
path = "Replica-Dataset/replica_v1/apartment_0/habitat/info_semantic.json"


target_instance_ID={
    "refrigerator": 67, 
    "rack": 66, 
    "cushion": 29, 
    "lamp": 47,
    "cooktop": 32
}


def transform(point):
    # (x,z) u,v
    # kitchen (-0.6427505, -1.5003817)  209,302
    # kitchen2 (1.7072494, 0.69961834)  272,235
    # wall corner (-3.0927505, 0.59961843) 275, 378
    # sofa corner (2.2072494, 9.099619)  595, 219
    # book corner (3.8072493, 5.749618)  435 171

    u, v = point

    scale_x=(2.2072494-1.7072494)/(219-235)
    scale_z=(0.69961834+1.5003817)/(272-209)

    x=(v-290)*scale_x
    z=(u-255)*scale_z

    return np.asarray([x,z])

# read searching target and path
search_path=[]
f=open(f"{args.data_path}/{args.target}_Path.txt","r")
for l in f.readlines():
    p=l[:-1].replace('(','').replace(')','').split(",")
    search_path+=[tuple([int(p[0]),int(p[1])])]
f.close()





# reset scale and start point
search_path=np.array(search_path)
p_start=transform(search_path[0])
p_start=np.array([p_start[0],-2.9252348,p_start[1]])

# p_start=np.zeros(3)

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img




def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.01) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
#agent_state.position = np.array([0.0, 0.0,0.0])  # agent in world space [x,y,z]
agent_state.position =p_start
print(agent_state.position)
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


for p in search_path:
    print(p,transform(p))
print("\n")
#raise Exception

def navigateAndSee(action="",writer=None):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        # highlight the target object
        RGB_img=transform_rgb_bgr(observations["color_sensor"])
        instance_label=id_to_label[observations["semantic_sensor"]]
        mask=(instance_label==target_instance_ID[args.target])

        if mask.any():
            RGB_img[mask,2]=255

        # show
        cv2.imshow("RGB", RGB_img)

        img=np.concatenate((map_with_bg,RGB_img),axis=1) 
        img=np.uint8(img)
        writer.write(img)
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        #cv2.imshow("semantic", transform_semantic(id_to_label[observations["semantic_sensor"]]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        #print("camera pose: x y z rw rx ry rz")
        #print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        https://blog.csdn.net/hankerbit/article/details/84066629
    """
    x1,y1 = unit_vector(v1)
    x2,y2 = unit_vector(v2)


    dot=x1*x1+y1*y2
    det=x1*y2-x2*y1
    
    rad= np.arctan2(det,dot)
    angle=rad*  180.0 / np.pi

    
    return angle

def get_forward_direction():
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    rw, rx, ry, rz=sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z
    forward_dir=np.array([
        -2 * (rx * rz + rw * ry),
        #2*(ry*rz-rw*rx), 
        2 * (rx * rx + ry * ry)-1,
                
              
        ])
    return forward_dir


def get_position():
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    return sensor_state.position[0],sensor_state.position[2]

# create video
fps = 40
size = (512+512, 512)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(f'./{args.target}.mp4', fourcc, fps, size)



action = "move_forward"
navigateAndSee(action,videowriter)


cv2.waitKey()
for i in range(1,search_path.shape[0]-1):

    current_dir=get_forward_direction()

    
    p1=transform(search_path[i])
    p2=np.array(get_position())
    #p2=transform(search_path[i-1])
    dx,dy=p1-p2
    dis=sqrt(dx**2+dy**2) 

    next_dir=np.array([dx,dy])

    print(f"from {p2} to {p1}")
    print(f"forward: {current_dir}\ntarget: {next_dir}")
    
    # caculate angle between 2 vector
    angle=angle_between(current_dir,next_dir)
    if angle<0:
        action="turn_left"
    else:
        action="turn_right"
    round_angle=round(angle)
    print(angle,round_angle,action)

    for t in range(abs(round_angle)):
        navigateAndSee(action,videowriter)
        cv2.waitKey(1)

    

    for t in range(round(dis/0.01)):
        navigateAndSee("move_forward",videowriter)
        cv2.waitKey(1)


    print(get_position())
    print("\n")

    '''keystroke = cv2.waitKey(0)
                if keystroke == ord(FINISH):
                    print("action: FINISH")
                    break'''





"""while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue"""
