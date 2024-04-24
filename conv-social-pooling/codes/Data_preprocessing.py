import numpy as np
import pandas as pd 
import pickle
import copy

class DataPoint():
	def __init__(self, data_array, reference_time, dataset_id, attention_distance = 90, grid_size = 15):
        
		data_point = data_array
		self.dataset_id = dataset_id 
		self.id = int(data_point[0])  
		self.time = np.round(int(data_point[1]), 1)
		self.x_loc = float(data_point[2])
		self.y_loc = float(data_point[3])
		self.lane = int(data_point[4])
		self.speed = float(data_point[5]) #feet/second
		self.acceleration = float(data_point[6]) #feet/second square  
		self.frame_number = 1

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

class Ngsim:
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
		lanes = pd.read_csv(self.file)['lane'].unique()
		self.lane_dict = {l:i for i in range(len(lanes)) for l in lanes} 
		self.process_points()

	def process_points(self):
		print("Processing all the points")
		df = pd.read_csv(self.file)  
		selected_columns = df.keys()[:7] 
		data_arrays = df[selected_columns].to_numpy() 
		self.data_points = [DataPoint(i, self.t_reference, self.dataset_id) for i in data_arrays] 
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
		print("All the points are processed and thre trajectories are constructed!")

	def find_location_grid(self, time_resolution, num_lanes):
		print("finding the location grid for each point ...")
		errors = 0
		times = [p.time for p in self.data_points]
		locations = [p.y_loc for p in self.data_points]
		min_time = min(times)
		max_time = max(times)
		min_location = min(locations)
		max_location = max(locations)
		grid_size = self.data_points[0].grid_size
		grid_shape = (num_lanes, int((max_location-min_location)/grid_size)+1)
		print("min_location: ", min_location)
		print("max_location: ", max_location)
		t_steps = int(round((max_time-min_time)/time_resolution,0))
		global_grid = [np.zeros(grid_shape, dtype=int) for s in range(t_steps+1)]
 
 
		for p in self.data_points: 
			t_ind = int(round(p.time/time_resolution, 0))  
			lane_ind = self.lane_dict[p.lane]
			location_ind = int(p.y_loc/grid_size)  

			if location_ind >= len(global_grid[t_ind][lane_ind]):
				errors+=1
			elif global_grid[t_ind][lane_ind][location_ind] == 0:
				global_grid[t_ind][lane_ind][location_ind] = p.id
			else: 
				errors += 1

		print("WARNING - total errors:", errors)
		print("Global Grid is constructed")

		
		for p in self.data_points:
			t_ind = int(round(p.time/time_resolution, 0))
			lane_ind = self.lane_dict[p.lane]
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
		print("output_history: ",  output_history)
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
		t_multiplier = int(round(1 / time_resolution))
		for trajectory in self.trajectories:
			for i in range(len(trajectory.points)):
				p = trajectory.points[i]
				# Lateral movement: considers lane changing 
				s_index = i
				e_index = min(len(trajectory.points) - 1, i + lateral_m_t_modified * t_multiplier)
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
				s_index = max(0, i - input_history * t_multiplier)
				e_index = min(len(trajectory.points)-1, i + output_history * t_multiplier)
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

def process_data_flow_density(file, dataset_id, reference_time = 0, input_history = 3, output_history = 5, speed_ratio_braking = 0.8, lateral_m_t = 4, time_resolution = 0.1, num_lanes = 8):
	print("output_history: ", output_history)
	ngsim_data = Ngsim(file, reference_time, dataset_id, input_history, output_history, speed_ratio_braking,
					   lateral_m_t, time_resolution, num_lanes)

	total_data = ngsim_data

	# getting flow and density
	if dataset_id in [1, 2, 3]:
		data_col_loc = "ngsim-101"
	elif dataset_id in [4, 5, 6]:
		data_col_loc = "ngsim-80"
	flow_list, density_list = flow_density_NGSIM_data(total_data, time_resolution, data_col_loc, fd_time_size=20)

	# Count the number of data points for which there is 3 second data history and at least one future point:
	num_useful_points = 0
	for trajectory in total_data.trajectories:
		if len(trajectory.points) < 32:
			continue
		else:
			num_useful_points += len(trajectory.points) - 30 - 1
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
		if len(traj.points) < 32: # 30 for history, 1 for current point, 1 for future point
			continue
		else:
			# for i in range(30, len(traj.points) - 1): #keep at least one point for prediction
			for i in range(30, len(traj.points) - 1): #keep at least one point for prediction
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
				point.append(flow_list[t_step])
				point.append(density_list[t_step])

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
					neighbor_traj = dataset_trajectories[neighbor_id - 1] 
					neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
					po_x.append(neighbor_pos[0])
					po_y.append(neighbor_pos[1])

			for n in range(len(t_p.neighbors_same_lane)):
				neighbor_id = t_p.neighbors_same_lane[n]
				if neighbor_id == 0:
					po_x.append(0)
					po_y.append(0)
				else: 
					neighbor_traj = dataset_trajectories[neighbor_id - 1] 
					neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
					po_x.append(neighbor_pos[0])
					po_y.append(neighbor_pos[1])

			for n in range(len(t_p.neighbors_right_lane)):
				neighbor_id = t_p.neighbors_right_lane[n]
				if neighbor_id == 0:
					po_x.append(0)
					po_y.append(0)
				else: 
					neighbor_traj = dataset_trajectories[neighbor_id - 1] 
					neighbor_pos = neighbor_traj[np.where(neighbor_traj[:, 0] == t_p.frame_number)][0, 1:3]
					po_x.append(neighbor_pos[0])
					po_y.append(neighbor_pos[1])

			points_x.append(po_x)
			points_y.append(po_y)
		points_x = np.array(points_x)
		points_y = np.array(points_y)
		dataset_trajectories_x_grid[trajectory.id - 1] = points_x
		dataset_trajectories_y_grid[trajectory.id - 1] = points_y
 
	 
	return(training_set, validation_set, testing_set, dataset_trajectories, dataset_trajectories_x_grid, dataset_trajectories_y_grid)


def flow_density_NGSIM_data(total_data, time_resolution, data_collection_location, fd_time_size = 20):
	#data_location can be "ngsim-101" or "ngsim-80"
	total_data.data_points.sort(key=lambda x: x.y_loc)
	min_y_loc = total_data.data_points[0].y_loc
	max_y_loc = total_data.data_points[-1].y_loc
	segment_length = round(max_y_loc - min_y_loc, 1) 

	total_data.data_points.sort(key=lambda x: x.time)
	min_time = total_data.data_points[0].time
	max_time = total_data.data_points[-1].time 
	total_data.data_points.sort(key=lambda x: x.id)

	total_time_steps = int(round((max_time - min_time) / time_resolution)) + 1 
	li_list = [0 for i in range(total_time_steps)]
	ti_list = [0 for i in range(total_time_steps)]

	for trajectory in total_data.trajectories:
		for i in range(len(trajectory.points)):
			if data_collection_location == "ngsim-101":
				if trajectory.points[i].lane > 5:
					continue
				else:
					num_lanes = 5
			elif data_collection_location == "ngsim-80":
				if trajectory.points[i].lane > 7:
					continue
				else:
					num_lanes = 6
			if i == 0:
				delta_y = 0
				delta_t = 0
			else:
				delta_y = trajectory.points[i].y_loc - trajectory.points[i - 1].y_loc
				delta_t = total_data.time_resolution
			t_step = int(round(trajectory.points[i].time / total_data.time_resolution))
			li_list[t_step] += delta_y
			ti_list[t_step] += delta_t

	li_list = np.array(li_list) / num_lanes
	ti_list = np.array(ti_list) / num_lanes
	flow_list = [0 for i in range(total_time_steps)]
	density_list = [0 for i in range(total_time_steps)]
	fd_steps = int(round(fd_time_size / time_resolution))
	for i in range(1, total_time_steps):
		if i - fd_steps >= 1:
			li_s = li_list[i - fd_steps:i + 1]
			ti_s = ti_list[i - fd_steps:i + 1]
		else:
			li_s = li_list[1:i + 1]
			ti_s = ti_list[1:i + 1]
		block_size = len(li_s) * time_resolution * segment_length
		flow = np.round(np.sum(li_s) / block_size * 3600, 1)  # to convert flow to vehicle per hour
		density = np.round(np.sum(ti_s) / block_size * 5280, 1)  # to convert density to vehicle per mile
		flow_list[i] = flow
		density_list[i] = density

	flow_list[0] = flow_list[1]
	density_list[0] = density_list[1]
	return flow_list, density_list


################################################### MAIN FUNCTION ################################################################################################
def main():
	file_directory = ""
	file_name = "I294_Cleaned.csv"
	output_directory = "cee497projects/trajectory-prediction/data/101-80-speed-maneuver-for-GT/10_seconds/"
	reference_time = 6
	dataset_id = 1
	input_history = 3
	output_history = 10
	speed_ratio_braking = 0.8
	lateral_m_t = 5
	time_resolution = 0.1
	
	mean_flow = 1133.49
	std_flow = 417.05
	mean_density = 68.53
	std_density = 24.67

	data = pd.read_csv(file_name) 
	num_lanes = len(data['lane'].unique()) 

	total_train_set = []
	total_validation_set = []
	total_test_set = []
	total_trajectories = []
	total_trajectories_x = []
	total_trajectories_y = [] 


	for i in range(6): 
		print(f'Working on Dataset id: {i+1}')
		
		file = file_directory + file_name
		training_set, validation_set, testing_set, dataset_trajectories, dataset_trajectories_x_grid, \
		dataset_trajectories_y_grid = process_data_flow_density(file, i+1, reference_time,
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
		 
		
		
	print("Saving files...")
	with open(output_directory+"train.data", 'wb') as filehandle:
		pickle.dump(total_train_set, filehandle)

	with open(output_directory+"valid.data", 'wb') as filehandle:
		pickle.dump(total_validation_set, filehandle)

	with open(output_directory+"test.data", 'wb') as filehandle:
		pickle.dump(total_test_set, filehandle)

	with open(output_directory+"train_trajectory.data", 'wb') as filehandle:
		pickle.dump(total_trajectories, filehandle)

	with open(output_directory+"train_trajectory_x.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_x, filehandle)

	with open(output_directory+"train_trajectory_y.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_y, filehandle)
	

	with open(output_directory+"valid_trajectory.data", 'wb') as filehandle:
		pickle.dump(total_trajectories, filehandle)

	with open(output_directory+"valid_trajectory_x.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_x, filehandle)

	with open(output_directory+"valid_trajectory_y.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_y, filehandle)
	
	with open(output_directory+"test_trajectory.data", 'wb') as filehandle:
		pickle.dump(total_trajectories, filehandle)

	with open(output_directory+"test_trajectory_x.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_x, filehandle)

	with open(output_directory+"test_trajectory_y.data", 'wb') as filehandle:
		pickle.dump(total_trajectories_y, filehandle)
	

	print("All files are saved!")


if __name__ == "__main__":
	main()