import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import math
import random as rand

# define pendulum dynamics function 
def pend(t, thetas, g, l):
    theta, theta_dot = thetas

    theta_ddot = -g/l*math.sin(theta)
    return [theta_dot, theta_ddot]

# initialize simulation parameters
traj_len = 21
start_time = 0
stop_time = .4
n_training_configs = 2
n_testing_configs = 1
n_training_ICs = 2
n_testing_ICs = 1
time = np.linspace(start_time, stop_time, traj_len)

# initialize trajectory data
training_data = np.zeros([n_training_configs*n_training_ICs*traj_len, 4])
testing_data = np.zeros([n_testing_configs*n_testing_ICs*traj_len, 4])

# initialize context data (array that stores parameters g and l)
train_configs = np.zeros([n_training_configs, 2])
test_configs = np.zeros([n_testing_configs, 2])

for i in range(n_training_configs):
        # get random parameters for g and l 
        g = rand.uniform(1, 12)
        l = rand.uniform(.2, 1)
        train_configs[i,:] = [g, l]

        # repeat and stack parameters for each trajectory in this sub-experiment
        repeated_params = np.tile([g,l], (n_training_ICs*traj_len,1))

        # get the starting and ending indices for each parametric configuration
        exp_starting_index = i*n_training_ICs*traj_len
        exp_ending_index = (i+1)*n_training_ICs*traj_len

        # assign parameters to data
        training_data[exp_starting_index:exp_ending_index, 2:] = repeated_params

        for traj in range(n_training_ICs):
            # get starting and ending indices for each trajectory
            traj_starting_index = exp_starting_index + traj*traj_len
            traj_ending_index = exp_starting_index + (traj+1)*traj_len

            # get random initial conditions
            theta_0 = rand.uniform(0, 2*math.pi)
            theta_dot_0 = rand.uniform(0, 2*math.pi)

            # simulate dynamics
            sol = solve_ivp(pend, t_span=[start_time, stop_time], y0=[theta_0, theta_dot_0], args=(g, l), t_eval=time)

            # get trajectory
            z = sol.y
            trajectory = z.T

            # assign trajectory to data
            training_data[traj_starting_index:traj_ending_index, 0:2] = trajectory

for i in range(n_testing_configs):
        # get random parameters for g and l 
        g = rand.uniform(1, 12)
        l = rand.uniform(.2, 1)
        test_configs[i,:] = [g, l]

        # repeat and stack parameters for each trajectory in this sub-experiment
        repeated_params = np.tile([g,l], (n_testing_ICs*traj_len,1))

        # get the starting and ending indices for each parametric configuration
        exp_starting_index = i*n_testing_ICs*traj_len
        exp_ending_index = (i+1)*n_testing_ICs*traj_len

        # assign parameters to data
        testing_data[exp_starting_index:exp_ending_index, 2:] = repeated_params

        for traj in range(n_testing_ICs):
            # get starting and ending indices for each trajectory
            traj_starting_index = exp_starting_index + traj*traj_len
            traj_ending_index = exp_starting_index + (traj+1)*traj_len

            # get random initial conditions
            theta_0 = rand.uniform(0, 2*math.pi)
            theta_dot_0 = rand.uniform(0, 2*math.pi)

            # simulate dynamics
            sol = solve_ivp(pend, t_span=[start_time, stop_time], y0=[theta_0, theta_dot_0], args=(g, l), t_eval=time)

            # get trajectory
            z = sol.y
            trajectory = z.T

            # assign trajectory to data
            testing_data[traj_starting_index:traj_ending_index,0:2] = trajectory



np.save("Pendulum_debug_train_data", training_data)
np.save("Pendulum_debug_train_configs", train_configs)

np.save("Pendulum_test_data", testing_data)
np.save("Pendulum_test_configs", testing_data)


