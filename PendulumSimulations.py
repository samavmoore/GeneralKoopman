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
traj_len = 201
start_time = 0
stop_time = 4
n_configs = 20
n_ICs
n_testing_ICs = 1


# initialize trajectory data

testing_data = np.zeros([n_testing_configs*n_testing_ICs*traj_len, 4])

# initialize context data (array that stores parameters g and l)

test_configs = np.zeros([n_testing_configs, 2])



def get_pendulum_data(n_ICs, n_configs, start_time, stop_time, traj_len):
    configs = np.zeros([n_configs, 2])
    data = np.zeros([n_configs*n_ICs*traj_len, 4])
    time = np.linspace(start_time, stop_time, traj_len)

    for i in range(n_configs):
            # get random parameters for g and l 
            g = rand.uniform(1, 12)
            l = rand.uniform(.1, 1)
            configs[i,:] = [g, l]

            # repeat and stack parameters for each trajectory in this sub-experiment
            repeated_params = np.tile([g,l], (n_ICs*traj_len,1))

            # get the starting and ending indices for each parametric configuration
            exp_starting_index = i*n_ICs*traj_len
            exp_ending_index = (i+1)*n_ICs*traj_len

            # assign parameters to data
            data[exp_starting_index:exp_ending_index, 2:] = repeated_params

            for traj in range(n_ICs):
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
                data[traj_starting_index:traj_ending_index, 0:2] = trajectory

    return configs, data 








if __name__=="__main__":

########----------- Context Pre-train Data ---------------------

    n_context_configs = 1000

    context = np.zeros([1000, 2])

    for i in range(n_context_configs):
        g = rand.uniform(1, 12)
        l = rand.uniform(.2, 1)
        context[i, :] = [g, l]

##########---------------- Main Training Data -----------------


########------------ Main Test/Validation Data ----------------



np.save("context_data", context)

#np.save("Pendulum_debug_train_data", training_data)
#np.save("Pendulum_debug_train_configs", train_configs)

#np.save("Pendulum_test_data", testing_data)
#np.save("Pendulum_test_configs", testing_data)


