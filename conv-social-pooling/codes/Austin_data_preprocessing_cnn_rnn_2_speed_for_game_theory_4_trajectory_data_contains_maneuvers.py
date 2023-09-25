__author__ = "Mohammadreza Khajeh-Hosseini"
import numpy as np
import pickle
import copy

class DataPoint():
    def __init__(self, data_array, reference_time, dataset_id, attention_distance = 90, grid_size = 15):
        data_point = data_array.split(",")
#         print(data_point)
        #ID,time,front-bumper,y,lane,speed,acceleration,leader-ID,leader-time,leader-front-bumper,leader-y,
        #leader-lane,leader-speed,leader-acceleration,length,width,type
        self.dataset_id = dataset_id
        self.id = int(data_point[0])
        self.frame_number = int(round(float(data_point[1])/0.08))+1
        self.time = float(data_point[1])
        self.x_loc = float(data_point[3])*3.28084 # from meter converted to feet
        self.y_loc = float(data_point[2])*3.28084 # from meter converted to feet
        self.lane = int(data_point[4]) + 1 #lane numbers originaly from 0-3, now it would be from 1-4
        self.speed = float(data_point[5])*3.28084 #feet/second
        self.acceleration = float(data_point[6])*3.28084 #feet/second square

        # Map of the surrounding vehicles in the grid form based on the attention_distance and grid_size
        self.attention_distance = attention_distance
        self.grid_size = grid_size
        self.grids_on_each_side = int(round(self.attention_distance/self.grid_size, 0))
        self.neighbors_left_lane = [0 for i in range(self.grids_on_each_side*2+1)]
        self.neighbors_right_lane = [0 for i in range(self.grids_on_each_side*2+1)]
        self.neighbors_same_lane = [0 for i in range(self.grids_on_each_side*2+1)]
        self.neighbors_same_lane[self.grids_on_each_side] = self.id

        #Maneuver:
        self.lateral_maneuver = 0
        self.longitudinal_maneuver = 0

class Trajectory():
    def __init__(self, id):
        self.id = id
        self.points = []

class Austin:
    def __init__(self, data_file, reference_time, dataset_id, input_history, output_history, speed_ratio_braking,
                 lateral_m_t, time_resolution, num_lanes):
        self.file = data_file
        self.t_reference = reference_time
        self.dataset_id = dataset_id
        self.input_history = input_history
        self.output_history = output_history
        self.speed_ratio_braking = speed_ratio_braking
        self.lateral_m_t = lateral_m_t
        self.time_resolution = time_resolution
        self.num_lanes = num_lanes
        self.process_points()

    def process_points(self):
        print("processing points ...")
        data_arrays = open(self.file ,'r').read().splitlines()
        self.data_points = [DataPoint(data_arrays[i], self.t_reference, self.dataset_id) for i in range(1,len(data_arrays))]
        print("total number of data points: ", len(self.data_points))
        self.data_points.sort(key = lambda x:x.id)
        self.max_id = self.data_points[-1].id
        self.min_id = self.data_points[0].id
        print("total number of points: ", len(self.data_points))
        print("maximum id: ", self.max_id)
        print("minimum id: ", self.min_id)
        self.find_location_grid(self.time_resolution, self.num_lanes)
        self.trajectories = [Trajectory(i) for i in range(self.max_id+1)]
        for p in self.data_points:
            self.trajectories[p.id].points.append(p)
        self.find_maneuver(self.input_history, self.output_history, self.speed_ratio_braking,
                           self.lateral_m_t, self.time_resolution)
        print("all the points are processed and thre trajectories are constructed!")

    def find_location_grid(self, time_resolution, num_lanes):
        print("finding the location grid for each point ...")
        errors = 0
        times = [p.time for p in self.data_points]
        locations = [p.y_loc for p in self.data_points]
        min_time = min(times)
        max_time = max(times)
        min_location = min(locations)
        max_location = max(locations)
        print("min(times): ", min(times))
        print(" max(times): ",  max(times))
        print("min_location: ", min_location)
        print("max_location: ", max_location)
        min_location = 0
        print("min_location is changed to 0 manually: ", min_location)
        grid_size = self.data_points[0].grid_size
        grid_shape = (num_lanes, int((max_location-min_location)/grid_size)+1)
        t_steps = int(round((max_time-min_time)/time_resolution,0))
        global_grid = [np.zeros(grid_shape, dtype=int) for s in range(t_steps+1)]

        for p in self.data_points:
            t_ind = int(round(p.time/time_resolution, 0))
            lane_ind = p.lane-1
            location_ind = int(p.y_loc/grid_size)

            if global_grid[t_ind][lane_ind][location_ind] == 0:
                global_grid[t_ind][lane_ind][location_ind] = p.id
            else:
                # print("ERROR: at time ", round(self.t_reference+t_ind*time_resolution,1), ", the vehicles ",
                # global_grid[t_ind][lane_ind][location_ind], " and ", p.id,
                # " are at the grid location at the same time!")
                errors += 1
        print("WARNING - total errors:", errors)
        print("global grid is constructed")

        for p in self.data_points:
            t_ind = int(round(p.time/time_resolution, 0))
            lane_ind = p.lane-1
            location_ind = int(p.y_loc/grid_size)

            for i in range(1,p.grids_on_each_side+1):
                # the location of the vehicle on its grid is at p.grids_on_each_side,
                # example: p.grids_on_each_side=6, [0,1,2,3,4,5,vehicle,7,8,9,10,11,12]

                # infront of the vehicle:
                if location_ind+i in range(grid_shape[1]):
                    #for the same lane:
                    p.neighbors_same_lane[p.grids_on_each_side+i] = global_grid[t_ind][lane_ind][location_ind+i]
                    #for the lane on the left:
                    if lane_ind-1 in range(grid_shape[0]):
                        p.neighbors_left_lane[p.grids_on_each_side+i] = \
                            global_grid[t_ind][lane_ind-1][location_ind+i]
                    #for the lane on the right:
                    if lane_ind+1 in range(grid_shape[0]):
                        p.neighbors_right_lane[p.grids_on_each_side + i] = \
                            global_grid[t_ind][lane_ind+1][location_ind+i]

                # behind the vehicle:
                if location_ind-i in range(grid_shape[1]):
                    #for the same lane:
                    p.neighbors_same_lane[p.grids_on_each_side-i] = global_grid[t_ind][lane_ind][location_ind-i]
                    #for the lane on the left:
                    if lane_ind-1 in range(grid_shape[0]):
                        p.neighbors_left_lane[p.grids_on_each_side-i] = \
                            global_grid[t_ind][lane_ind-1][location_ind-i]
                    #for the lane on the right:
                    if lane_ind+1 in range(grid_shape[0]):
                        p.neighbors_right_lane[p.grids_on_each_side-i] = \
                            global_grid[t_ind][lane_ind+1][location_ind-i]

        print("location grids: complete!")

    def find_maneuver(self, input_history, output_history, speed_ratio_braking, lateral_m_t_modified, time_resolution):
        """
        input_history: is the time duration of the input history of the trajectory
        output_history: is the time duration of the prediction output history
        speed_ratio_braking: if the ratio of average previous speed and the future speed are less than this ration
        then, we assume the vehicle is braking
        # NOT USED !!! - OLDER METHOD -lateral_m_t: is the duration of time considered for before and afre completion of the lateral movement
        lateral_m_t_modified: equal to prediction horizon or duration considered for after completion of the lateral movement
        time_resolution: is the time resolution of the data points which is 0.1 second in most cases

        """
        print("finding the type of longitudinal and lateral maneuvers for each point ...")
        # lateral maneuver within +- leteral_m_t seconds
        t_multiplier = round(1 / time_resolution,2)

        for trajectory in self.trajectories:
            for i in range(len(trajectory.points)):
                p = trajectory.points[i]
                # Lateral movement: considers lane changing
                #s_index = max(0, i-int(round(lateral_m_t*t_multiplier)))
                #e_index = min(len(trajectory.points) - 1, i + int(round(lateral_m_t * t_multiplier)))
                s_index = i
                e_index = min(len(trajectory.points) - 1, i + int(round(lateral_m_t_modified * t_multiplier)))
                if trajectory.points[e_index].lane > p.lane or trajectory.points[s_index].lane < p.lane:
                    #vehicle has moved right
                    p.lateral_maneuver = 3
                elif trajectory.points[e_index].lane < p.lane or trajectory.points[s_index].lane > p.lane:
                    #vehicle has moves left
                    p.lateral_maneuver = 2
                else:
                    #no lateral move
                    p.lateral_maneuver = 1

                # Longitudinal movement: considers braking
                s_index = max(0, i - int(round(input_history * t_multiplier)))
                e_index = min(len(trajectory.points)-1, i + int(round(output_history * t_multiplier)))
                if e_index == i or s_index == i:
                    #no braking
                    p.longitudinal_maneuver = 1
                else:
                    v_input_history = (p.y_loc - trajectory.points[s_index].y_loc)/(i - s_index)
                    v_output_history = (trajectory.points[e_index].y_loc - p.y_loc)/(e_index - i)
                    if v_output_history/(v_input_history+0.0001) < 0.8:
                        #braking
                        p.longitudinal_maneuver = 2
                    else:
                        #no braking
                        p.longitudinal_maneuver = 1

        print("finding the maneuvers is complete!")


def process_data_flow_density(file, dataset_id, reference_time = 0, input_history = 3.2, output_history = 5.12, speed_ratio_braking = 0.8, lateral_m_t = 5.12, time_resolution = 0.1, num_lanes = 4):

    austin_data = Austin(file, reference_time, dataset_id, input_history, output_history, speed_ratio_braking,
                         lateral_m_t, time_resolution, num_lanes)

    total_data = austin_data

    # getting flow and density
    if dataset_id in [1]:
        data_col_loc = "Austin"

    flow_list = []
    density_list = []
    frames_list = []

    # Count the number of data points for which there is 3.2 second data history and at least one future point:
    num_useful_points = 0
    for trajectory in total_data.trajectories:
        if len(trajectory.points) < 42:
            continue
        else:
            num_useful_points += len(trajectory.points) - 40 - 1
    print("total number of useful points: ", num_useful_points)
    num_training = int(0.7 * num_useful_points)
    num_validation = int(0.1 * num_useful_points)
    num_testing = num_useful_points - num_training - num_validation
    print("points for training, validation, and testing respectively: ", num_training, num_validation, num_testing)

    training_set = []
    validation_set = []
    testing_set = []
    counter = 1

    for traj in total_data.trajectories:
        if len(traj.points) < 42: # 40 for history, 1 for current point, 1 for future point
            continue
        else:
            # for i in range(40, len(traj.points) - 1): #keep at least one point for prediction
            for i in range(40, len(traj.points) - 1): #keep at least one point for prediction
                p = traj.points[i]
                point = [p.dataset_id, p.id, p.frame_number, p.x_loc, p.y_loc, p.lane, p.lateral_maneuver,
                         p.longitudinal_maneuver]
                for n in range(len(p.neighbors_left_lane)):
                    point.append(p.neighbors_left_lane[n])
                for n in range(len(p.neighbors_same_lane)):
                    point.append(p.neighbors_same_lane[n])
                for n in range(len(p.neighbors_right_lane)):
                    point.append(p.neighbors_right_lane[n])

                #Add flow and density
                t_step = int(round(traj.points[i].time/time_resolution))
                point.append(0) # no flow or density data is passed
                point.append(0) # no flow or density data is passed

                if counter <= num_training:
                    training_set.append(point)
                elif counter <= num_training + num_validation:
                    validation_set.append(point)
                else:
                    testing_set.append(point)
                counter += 1

    max_id = max([p.id for p in total_data.trajectories])
    min_id = min([p.id for p in total_data.trajectories])
    dataset_trajectories = [[] for i in range(max_id + 1)]
    for trajectory in total_data.trajectories:
        points = []
        for t_p in trajectory.points:
            po = [t_p.frame_number, t_p.x_loc, t_p.y_loc]
            for n in range(len(t_p.neighbors_left_lane)):
                po.append(t_p.neighbors_left_lane[n])
            for n in range(len(t_p.neighbors_same_lane)):
                po.append(t_p.neighbors_same_lane[n])
            for n in range(len(t_p.neighbors_right_lane)):
                po.append(t_p.neighbors_right_lane[n])

            #ADDED FOR GAME THEORY TRAINING
            po.append(t_p.dataset_id)
            po.append(t_p.lane)
            po.append(t_p.lateral_maneuver)
            po.append(t_p.longitudinal_maneuver)
            po.append(t_p.speed)
            po.append(t_p.acceleration)
            points.append(po)
        points = np.array(points)
        dataset_trajectories[trajectory.id - 1] = points
        # dataset_trajectories[trajectory.id - 1] = points.transpose()

    # These two sets of trajectory contain grids with x and y locations instead of vehicle id
    dataset_trajectories_x_grid = [[] for i in range(max_id + 1)]
    dataset_trajectories_y_grid = [[] for i in range(max_id + 1)]

    for trajectory in total_data.trajectories:
        points_x = []
        points_y = []
        for t_p in trajectory.points:
            po_x = [t_p.frame_number, t_p.x_loc, t_p.y_loc]
            po_y = [t_p.frame_number, t_p.x_loc, t_p.y_loc]
            for n in range(len(t_p.neighbors_left_lane)):
                neighbor_id = t_p.neighbors_left_lane[n]
                if neighbor_id == 0:
                    po_x.append(0)
                    po_y.append(0)
                else:
                    # neighbor_traj = dataset_trajectories[neighbor_id - 1].transpose()
                    neighbor_traj = dataset_trajectories[neighbor_id - 1]
                    #print("LEFT-neighbor trajectoy: ", neighbor_traj)
                    # print("t_p.frame_number: ", t_p.frame_number)
                    # print("neighbor_traj: ", neighbor_traj)
                    # print("neighbor id: ", neighbor_id, ", len neighbor traj: ", len(neighbor_traj))
                    # print("where matches frame number", neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)])
                    neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
                    po_x.append(neighbor_pos[0])
                    po_y.append(neighbor_pos[1])

            for n in range(len(t_p.neighbors_same_lane)):
                neighbor_id = t_p.neighbors_same_lane[n]
                if neighbor_id == 0:
                    po_x.append(0)
                    po_y.append(0)
                else:
                    # neighbor_traj = dataset_trajectories[neighbor_id - 1].transpose()
                    neighbor_traj = dataset_trajectories[neighbor_id - 1]
                    #print("SAME-neighbor trajectoy: ", neighbor_traj)
                    # print("t_p.frame_number: ", t_p.frame_number)
                    # print("neighbor_traj: ", neighbor_traj)
                    # print("neighbor id: ", neighbor_id, ", len neighbor traj: ", len(neighbor_traj))
                    # print("where matches frame number", neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)])
                    neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
                    po_x.append(neighbor_pos[0])
                    po_y.append(neighbor_pos[1])

            for n in range(len(t_p.neighbors_right_lane)):
                neighbor_id = t_p.neighbors_right_lane[n]
                if neighbor_id == 0:
                    po_x.append(0)
                    po_y.append(0)
                else:
                    # neighbor_traj = dataset_trajectories[neighbor_id - 1].transpose()
                    neighbor_traj = dataset_trajectories[neighbor_id - 1]
                    #print("RIGHT-neighbor trajectoy: ", neighbor_traj)
                    # print("t_p.frame_number: ", t_p.frame_number)
                    # print("neighbor_traj: ", neighbor_traj)
                    # print("neighbor id: ", neighbor_id, ", len neighbor traj: ", len(neighbor_traj))
                    # print("matches frame number: ", neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)])
                    neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
                    po_x.append(neighbor_pos[0])
                    po_y.append(neighbor_pos[1])

            points_x.append(po_x)
            points_y.append(po_y)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        dataset_trajectories_x_grid[trajectory.id - 1] = points_x
        dataset_trajectories_y_grid[trajectory.id - 1] = points_y

    #Create flow density non-normalized
    flow_density_frame = []
    for i in range(len(flow_list)):
        flow_density_frame.append([flow_list[i], density_list[i], frames_list[i]])
    flow_density_frame = np.array(flow_density_frame)

    return(training_set, validation_set, testing_set, dataset_trajectories, dataset_trajectories_x_grid, dataset_trajectories_y_grid, flow_density_frame)


#_______________________________________________________________________________________________________________________
# file_directory = '/data/reza/projects/trajectory-prediction/data/Austin/'
# '/data/reza/projects/trajectory-prediction/data/Austin/'

file_directory = '/'

file_dict = {1: "trajectories_kf_filter_smoothed_front_bumper_with_leader.csv"}
output_directory = "/data/reza/projects/trajectory-prediction/data/Austin/5120_miliseconds/"
ref_time_dict = {1: 0}
reference_time = 0
dataset_id = 1
input_history = 3.2
output_history = 5.12
speed_ratio_braking = 0.8
lateral_m_t = 5
time_resolution = 0.08
num_lanes = 5
#_______________________________________________________________________________________________________________________

total_train_set = []
total_validation_set = []
total_test_set = []
total_trajectories = []
total_trajectories_x = []
total_trajectories_y = []
total_flow_density_frame = []

# for i in range(6):
for i in range(1):
	print("___________________")
	print("Working on datased id: ", i+1)
	file_name = file_dict[(i+1)]
	file = file_directory + file_name
	reference_time = ref_time_dict[(i+1)]
	training_set, validation_set, testing_set, dataset_trajectories, dataset_trajectories_x_grid, \
	dataset_trajectories_y_grid, flow_density_frame = process_data_flow_density(file, i+1, reference_time,
																				input_history, output_history,
																				speed_ratio_braking, lateral_m_t,
																				time_resolution, num_lanes)
	for p in training_set:
		total_train_set.append(p)
	for p in validation_set:
		total_validation_set.append(p)
	for p in testing_set:
		total_test_set.append(p)
	total_trajectories.append(dataset_trajectories)
	total_trajectories_x.append(dataset_trajectories_x_grid)
	total_trajectories_y.append(dataset_trajectories_y_grid)
	print("___________________")

print("Saving files...")
with open(output_directory+"train.data", 'wb') as filehandle:
	pickle.dump(total_train_set, filehandle)

with open(output_directory+"valid.data", 'wb') as filehandle:
	pickle.dump(total_validation_set, filehandle)

with open(output_directory+"test.data", 'wb') as filehandle:
	pickle.dump(total_test_set, filehandle)

with open(output_directory+"train_trajectory.data", 'wb') as filehandle:
 	pickle.dump(np.array(total_trajectories), filehandle)

with open(output_directory+"train_trajectory_x.data", 'wb') as filehandle:
 	pickle.dump(np.array(total_trajectories_x), filehandle)

with open(output_directory+"train_trajectory_y.data", 'wb') as filehandle:
 	pickle.dump(np.array(total_trajectories_y), filehandle)

print("All files are saved!")