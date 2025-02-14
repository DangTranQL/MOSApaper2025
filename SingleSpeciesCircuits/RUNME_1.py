# -------------- PART 0: PYTHON PRELIM --------------


import importlib
import os
import time
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
import pyvista as pv
import gc

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from math import inf
from numpy.random import seed
from scipy.spatial import ConvexHull, Delaunay


# -------------- PART 0a: CHOOSE CIRCUIT AND SET UP FOLDER --------------


# Choose circuit
circuit = input("Please enter name of the circuit: ")

# Import circuit config file
config = importlib.import_module(circuit)

# Define the subfolder name
folder_name = f"MOSA_{circuit}"

# Create folder if not yet exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Jump to folder
os.chdir(folder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0b: DEFINE DYNAMICAL SYSTEM --------------


# dx/dt
Equ1 = config.Equ1
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    alpha = params[0]
    n     = params[1]
    val0 = Equ1(x, alpha, n)
    return np.array([val0])

# Define initial time
t = 0.0

# Define number of steady states expected
numss = int(input("Do you expect 1 or 2 stable steady states in your search space? Please enter either 1 or 2: "))


# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------


# Define analytical sensitivity expressions
S_alpha_xss_analytic = config.S_alpha_xss_analytic
S_n_xss_analytic = config.S_n_xss_analytic


# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------


# Print prompt
print("""
Only two sensitivity functions are present:
0. |S_alpha_xss|
1. |S_n_xss|
MOSA will anneal this pair.
""")

# Choose pair of functions
choice1 = 0
choice2 = 1

# List of sensitivity function names
sensitivity_labels = [
    "|S_alpha_xss|",
    "|S_n_xss|"]

# Save function names for later use
label1 = sensitivity_labels[choice1]
label2 = sensitivity_labels[choice2]


# -------------- PART 0e: CHANGING DIRECTORIES --------------


# Define the subfolder name
subfolder_name = f"MOSA_sensfuncs_{choice1}_and_{choice2}"

# Create folder if not yet exist
if not os.path.exists(subfolder_name):
    os.makedirs(subfolder_name)

# Jump to folder
os.chdir(subfolder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0f: DEFINE FUNCTIONS --------------

# DEFINE FUNCTION TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def euclidean_distance(x1, x2):
    return abs(x1 - x2)

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS AND YSS GIVEN SOME INITIAL GUESS
def ssfinder(alpha_val,n_val):
        
    # -------------------------------------------------------------------------------------------------------------------
    # If we have one steady state                                                                                       #|
    if numss == 1:                                                                                                      #|
                                                                                                                        #|
        # Create an empty numpy array                                                                                   #|
        xss1 = np.array([])                                                                                             #|
                                                                                                                        #|
        # Define initial guesses                                                                                        #|
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)                                                 #|
                                                                                                                        #|
        # Define array of parameters                                                                                    #|
        params = np.array([alpha_val, n_val])                                                                           #|
        is_valid = False                                                                                                                #|
        # For each until you get one that gives a solution or you exhaust the list                                      #|
        for InitGuess in InitGuesses:                                                                                   #|
                                                                                                                        #|
            # Get solution details                                                                                      #|
            output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)      #|
            xss = output                                                                                                #| If we inputted 1 
            fvec = infodict['fvec']                                                                                     #| for numss prompt
                                                                                                                        #|
            # Check if stable attractor point                                                                           #|
            delta = 1e-8                                                                                                #|
            dEqudx = (Equs(xss+delta, t, params)-Equs(xss, t, params))/delta                                            #|
            jac = np.array([[dEqudx]])                                                                                  #|
            eig = jac                                                                                                   #|
            instablility = np.real(eig) >= 0                                                                            #|
                                                                                                                        #|
            # Check conditions for valid steady states                                                                  #|
            # i.e. xss is nonzero, residuals small, and successful convergence                                          #|
            if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:                    #|
                # If valid solution, store it                                                                               #|
                xss1 = np.append(xss1,xss)                                                                             #|
                # Stop as we now have the one solution we need                                                          #|
                is_valid = True
                break                                                                                                   #|
        if is_valid == False:                                                                                                                #|
            # If not valid, set to nan                                                                                      #|
            xss1 = np.append(xss1,float('nan'))                                                                             #|
#            print(xss1)
        return xss1
    # -------------------------------------------------------------------------------------------------------------------
        
    # -------------------------------------------------------------------------------------------------------------------
    # If we have two steady states                                                                                      #|
    if numss == 2:                                                                                                      #|
                                                                                                                        #|
        # Create an empty numpy array                                                                                   #|
        xss1 = np.array([])                                                                                             #|
        xss2 = np.array([])                                                                                             #|
                                                                                                                        #|
        # Define initial guesses                                                                                        #|
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)                                                 #|
                                                                                                                        #|
        # Define array of parameters                                                                                    #|
        params = np.array([alpha_val, n_val])                                                                           #|
                                                                                                                        #|
        # To store valid solutions                                                                                      #|
        solutions = []                                                                                                  #|
                                                                                                                        #|
        # For each until you get one that gives a solution or you exhaust the list                                      #|
        for InitGuess in InitGuesses:                                                                                   #|
                                                                                                                        #|
            # Get solution details                                                                                      #|
            output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)      #|
            xss = output                                                                                                #|
            fvec = infodict['fvec']                                                                                     #|
                                                                                                                        #|
            # Check if stable attractor point                                                                           #|
            delta = 1e-8                                                                                                #| If we inputted 2
            dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta                                #| for numss prompt
            jac = np.array([[dEqudx]])                                                                                  #|
            eig = jac                                                                                                   #|
            instablility = np.real(eig) >= 0                                                                            #|
                                                                                                                        #|
            # Check conditions for valid steady states                                                                  #|
            # i.e. xss is nonzero, residuals small, and successful convergence                                          #|
            if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:                    #|
                # If this is the first valid solution, just store it                                                    #|
                if len(solutions) == 0:                                                                                 #|
                    solutions.append(xss)                                                                               #|
                else:                                                                                                   #|
                    # Compare the new solution with the previous one                                                    #|
                    if all(euclidean_distance(existing_x, xss) > DISTANCE_THRESHOLD for existing_x in solutions):       #|
                        solutions.append(xss)                                                                           #|
                        # Stop as we now have two distinct solutions                                                    #|
                        break                                                                                           #|
                                                                                                                        #|
        # After looping through the guesses, store the solutions or NaN if no distinct solutions were found             #|
        if len(solutions) == 2:                                                                                         #|
            # Two distinct solutions found, sort and store them                                                         #|
            solutions.sort()                                                                                            #|
            xss1 = np.append(xss1,solutions[0])                                                                         #|
            xss2 = np.append(xss2,solutions[1])                                                                         #|
        elif len(solutions) == 1:                                                                                       #|
            # Only one distinct solution found, store it twice                                                          #|
            xss1 = np.append(xss1,solutions[0])                                                                         #|
            xss2 = np.append(xss2,solutions[0])                                                                         #|
        else:                                                                                                           #|
            # No valid solutions found, store NaN                                                                       #|
            xss1 = np.append(xss1,float('nan'))                                                                         #|
            xss2 = np.append(xss2,float('nan'))                                                                         #|
        return xss1, xss2
    # -------------------------------------------------------------------------------------------------------------------
    

# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(xss_list, alpha_list, n_list, choice1, choice2):
    
    # Evaluate sensitivities
    S_alpha_xss = S_alpha_xss_analytic(xss_list, alpha_list, n_list)
    S_n_xss     = S_n_xss_analytic(xss_list, alpha_list, n_list)

    # Sensitivity dictionary
    sensitivities = {
        "S_alpha_xss": S_alpha_xss,
        "S_n_xss": S_n_xss}

    # Map indices to keys
    labels = {
        0: "S_alpha_xss",
        1: "S_n_xss"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]


# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):
	
	# Update parameter set
    alpha_val = solution["alpha"]
    n_val = solution["n"]

    # Create an empty numpy array
    xss_collect = np.array([])

    # Find steady states and store
    xss = ssfinder(alpha_val,n_val)
    xss_collect = np.append(xss_collect,xss)
    
    # Get sensitivity pair
    sens1, sens2 = senpair(xss_collect, solution["alpha"], solution["n"], choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    return ans1, ans2
    

# -------------- PART 1: GAUGING MOSA PARAMETERS --------------

# Sample alpha values
alpha_min = float(input("Please enter minimum alpha value: "))
alpha_max = float(input("Please enter maximum alpha value: "))
alpha_sampsize = int(input("Please enter the number of alpha samples: "))
alpha_samps = np.linspace(alpha_min, alpha_max, alpha_sampsize)

# Sample n values
n_min = float(input("Please enter minimum n value: "))
n_max = float(input("Please enter maximum n value: "))
n_sampsize = int(input("Please enter the number of n samples: "))
n_samps = np.linspace(n_min, n_max, n_sampsize)

# Create empty arrays to store corresponding values of xss and yss
xss_samps = np.array([])
sens1_samps = np.array([])
sens2_samps = np.array([])

# For each combination of parameters
for i in alpha_samps:
    for j in n_samps:
        
        # Get steady states and store
        xss = ssfinder(i,j)
        xss_samps = np.append(xss_samps,xss)
        # Get sensitivities and store
        sens1, sens2 = senpair(xss, i, j, choice1, choice2)
        sens1_samps = np.append(sens1_samps,sens1)
        sens2_samps = np.append(sens2_samps,sens2)

# Get min and max of each sensitivity and print
sens1_samps_min = np.nanmin(sens1_samps)
print("Min sampled value of first sensitivity function: ", sens1_samps_min)
sens2_samps_min = np.nanmin(sens2_samps)
print("Min sampled value of second sensitivity function: ", sens2_samps_min)
sens1_samps_max = np.nanmax(sens1_samps)
print("Max sampled value of first sensitivity function: ", sens1_samps_max)
sens2_samps_max = np.nanmax(sens2_samps)
print("Max sampled value of second sensitivity function: ", sens2_samps_max)

# Get MOSA energies
deltaE_sens1 = sens1_samps_max - sens1_samps_min
print("Sampled energy difference in 1st sensitivity function: ", deltaE_sens1)
deltaE_sens2 = sens2_samps_max - sens2_samps_min
print("Sampled energy difference in 2nd sensitivity function: ", deltaE_sens2)
deltaE = np.linalg.norm([deltaE_sens1, deltaE_sens2])
print("Sampled cumulative energy difference: ", deltaE)

# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.9): "))
temp_hot = deltaE / np.log(1/probability_hot)
print("Hot run temp = ", temp_hot)

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.01): "))
temp_cold = deltaE / np.log(1/probability_cold)
print("Cold run temp = ", temp_cold)


# -------------- PART 2a: MOSA PREPARATIONS --------------

# Print prompt
print("Now preparing to MOSA...")

# Get number of MOSA runs
runs = int(input("Number of MOSA runs you would like to complete (if in doubt enter 5): "))
print("MOSA runs = ", runs)

# For each run
for run in range(runs):
    print(f"MOSA run number: {run}")
    
    # Define lists to collect Pareto-optimal sensitivity and parameter values from each MOSA run
    pareto_Salpha = []
    pareto_Sn     = []
    pareto_alpha  = []
    pareto_n      = []
    
    # Delete archive and checkpoint json files
    files_to_delete = ["archive.json", "checkpoint.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

	# -------------- PART 2b: ANNEAL TO GET PARETO FRONT IN SENSITIVITY SPACE --------------
	
	# Set random number generator to 0
    seed(0)
	
	# Initialisation of MOSA
#    opt = mosa.Anneal()
#    opt.archive_size = 5000
#    opt.maximum_archive_rejections = opt.archive_size
#    opt.population = {"alpha": (alpha_min, alpha_max), "n": (n_min, n_max)}
    opt=mosa.Anneal()
    opt.archive_size=5000
    opt.maximum_archive_rejections=5000
    opt.population={"alpha":(0.01,50),"n":(0.01,10)}
	
	# Hot run options
#    opt.initial_temperature = temp_hot
#    opt.number_of_iterations = 100
#    opt.temperature_decrease_factor = 0.9
#    no_of_steps_from_hot_to_cold = int(np.ceil((temp_hot-temp_cold)/opt.temperature_decrease_factor))
#    opt.number_of_temperatures = no_of_steps_from_hot_to_cold
#    opt.number_of_solution_elements = {"alpha":1, "n":1}
#    step_scaling = 1/opt.number_of_iterations
#    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
    opt.initial_temperature=500
    opt.number_of_iterations=200
    opt.number_of_temperatures=200
    opt.temperature_decrease_factor=0.95
    opt.number_of_solution_elements={"alpha":1,"n":1}
    opt.mc_step_size={"alpha":1,"n":1}
    	
    # Hot run
    start_time = time.time()
    opt.evolve(fobj)
    print(f"Hot run time: {time.time() - start_time} seconds")
	
    # Cold run options
#    opt.initial_temperature = temp_cold
#    opt.number_of_iterations = 100
#    opt.number_of_temperatures = 100
#    opt.temperature_decrease_factor = 0.9
#    opt.number_of_solution_elements = {"alpha":1,"n":1}
#    step_scaling = 1/opt.number_of_iterations
#    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
    opt.initial_temperature=1
    opt.number_of_iterations=200
    opt.number_of_temperatures=200
    opt.temperature_decrease_factor=0.95
    opt.number_of_solution_elements={"alpha":1,"n":1}
    opt.mc_step_size={"alpha":0.1,"n":0.1}
	
    # Cold run
    start_time = time.time()
    opt.evolve(fobj)
    print(f"Cold run time: {time.time() - start_time} seconds")
    
    # Output 
    start_time = time.time()
    pruned = opt.prunedominated()
    print(f"Pruning time: {time.time() - start_time} seconds")
	
	# -------------- PART 2c: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["alpha"] for solution in data["Solution"]])
    print(f"Archive length: {length}")
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_Salpha.append(dummy1)
        pareto_Sn.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'pruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
	
    # -------------- PART 2d: STORE AND SAVE CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract alpha and n values from the solutions
    alpha_values = [solution["alpha"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(alpha_values, n_values):
        pareto_alpha.append(dummy1)
        pareto_n.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(alpha_values, n_values)
    plt.xlabel('alpha')
    plt.ylabel('n')
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'pruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()

    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save S_a pareto values
    filename = f"pareto_Salpha_run{run}.npy"
    np.save(filename,pareto_Salpha)
    # Save S_n pareto values
    filename = f"pareto_Sn_run{run}.npy"
    np.save(filename,pareto_Sn)
    # Save a pareto values
    filename = f"pareto_alpha_run{run}.npy"
    np.save(filename,pareto_alpha)
    # Save n pareto values
    filename = f"pareto_n_run{run}.npy"
    np.save(filename,pareto_n)

# -------------- PART 2f: COMBINE PARETO DATA --------------

# Combine and save Pareto Salpha data

pareto_Salpha_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_Salpha_run{run}.npy"
    pareto_Salpha = np.load(filename)  
    pareto_Salpha_combined = np.concatenate((pareto_Salpha_combined, pareto_Salpha))

save_filename = "pareto_Salpha_combined.npy"
np.save(save_filename, pareto_Salpha_combined)

del pareto_Salpha_combined, pareto_Salpha
gc.collect()

# Combine and save Pareto Sn data

pareto_Sn_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_Sn_run{run}.npy"
    pareto_Sn = np.load(filename)  
    pareto_Sn_combined = np.concatenate((pareto_Sn_combined, pareto_Sn))

save_filename = "pareto_Sn_combined.npy"
np.save(save_filename, pareto_Sn_combined)

del pareto_Sn_combined, pareto_Sn
gc.collect()

# Combine and save Pareto alpha data

pareto_alpha_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_alpha_run{run}.npy"
    pareto_alpha = np.load(filename)  
    pareto_alpha_combined = np.concatenate((pareto_alpha_combined, pareto_alpha))

save_filename = "pareto_alpha_combined.npy"
np.save(save_filename, pareto_alpha_combined)

del pareto_alpha_combined, pareto_alpha
gc.collect()

# Combine and save Pareto n data

pareto_n_combined = np.empty((0,))
for run in range(runs):
    filename = f"pareto_n_run{run}.npy"
    pareto_n = np.load(filename)  
    pareto_n_combined = np.concatenate((pareto_n_combined, pareto_n))

save_filename = "pareto_n_combined.npy"
np.save(save_filename, pareto_n_combined)

del pareto_n_combined, pareto_n
gc.collect()


# -------------- PART 3: CUMULATIVE NEW PARETO OPTIMAL POINTS --------------

# 3a: LOAD DATA

filename = "pareto_Salpha_combined.npy"
pareto_Salpha_combined = np.load(filename)

filename = "pareto_Sn_combined.npy"
pareto_Sn_combined = np.load(filename)

filename = "pareto_alpha_combined.npy"
pareto_alpha_combined = np.load(filename)

filename = "pareto_n_combined.npy"
pareto_n_combined = np.load(filename)

# 3b: PLOT CUMULATIVE SENSITIVITY SPACE

plt.figure(figsize=(5,5))
plt.scatter(pareto_Salpha_combined, pareto_Sn_combined, s=10)
plt.xlabel(r'$|S_{a}(x_{ss})|$')
plt.ylabel(r'$|S_{n}(x_{ss})|$')
plt.grid(True)
plt.title(f'Negative Autoregulation: Cumulative Pareto Front from {runs} Runs')
plt.savefig(f'cumulative_pareto_sensitivities.png', dpi=300)
plt.close()

# 3c: PLOT CUMULATIVE PARAMETER SPACE

plt.figure(figsize=(5,5))
plt.scatter(pareto_alpha_combined, pareto_n_combined, s=10)
plt.xlabel(r'$a$')
plt.ylabel(r'$n$')
plt.grid(True)
plt.title(f'Negative Autoregulation: Corresponding Parameters from {runs} Runs')
plt.savefig(f'cumulative_pareto_parameters.png', dpi=300)
plt.close()


# -------------- PART 4: REDUCED GRID SEARCH --------------

# 4a: FIND RECTANGULAR PRISM BOUNDS

# Define scattered points in 2D sensitivity space
points = np.array(list(zip(pareto_alpha_combined, pareto_n_combined)))

# Define rectangule to bound scatter
min_vals = np.min(points, axis=0)
max_vals = np.max(points, axis=0)

print(min_vals)
print(max_vals)

# 4b: COMPARE OLD VS NEW PARAM SPACE AREA

# Volume of the bounding rectangular prism
new_param_area = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1])
print(new_param_area)
# Volume of original parameter space
old_param_area = (alpha_max-alpha_min) * (n_max-n_min)
print(old_param_area)
# Volume reduction as percentage
percentage = (new_param_area / old_param_area) * 100
# Print percentage
print(f"New parameter space is {percentage:.2f}% of original parameter space volume.")

# 4c: SAMPLE WITHIN NEW PARAM SPACE WITH SAME DENSITY AS ORIGINAL GRID SEARCH

# Create a grid of evenly spaced points from old parameter space
a_density = 5000
n_density = 5000
a_vals = np.linspace(alpha_min,alpha_max,a_density)
n_vals = np.linspace(n_min,n_max,n_density)
grid_x, grid_y = np.meshgrid(a_vals,n_vals)
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Define Bounding Rectangle Manually
min_x, min_y = min_vals
max_x, max_y = max_vals
bounding_box = np.array([
    [min_x, min_y],  # Bottom-left
    [max_x, min_y],  # Bottom-right
    [max_x, max_y],  # Top-right
    [min_x, max_y],  # Top-left
    [min_x, min_y]   # Close the rectangle
])

# Filter points inside the bounding rectangle
inside_rect_mask = (
    (grid_points[:, 0] >= min_x) & (grid_points[:, 0] <= max_x) &
    (grid_points[:, 1] >= min_y) & (grid_points[:, 1] <= max_y)
)
inside_rect_points = grid_points[inside_rect_mask]

# Save data
np.save("inside_points.npy", inside_rect_points)


# 4d: Plot bounding box

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Scatter plot of sampled points
ax.scatter(inside_rect_points[:, 0], inside_rect_points[:, 1], 
           s=1, color="blue", alpha=0.3, label="Sampled Points")

# Plot the bounding rectangle
bounding_box = np.array([
    [min_x, min_y],  # Bottom-left
    [max_x, min_y],  # Bottom-right
    [max_x, max_y],  # Top-right
    [min_x, max_y],  # Top-left
    [min_x, min_y]   # Close the rectangle
])
ax.plot(bounding_box[:, 0], bounding_box[:, 1], color="red", linewidth=2, label="Bounding Rectangle")

# Set labels and title
ax.set_xlabel("Alpha")
ax.set_ylabel("N")
ax.set_title("Bounding Rectangle and Sampled Points")
ax.set_xlim([alpha_min,alpha_max])
ax.set_ylim([n_min,n_max])

# Add legend
ax.legend()

# Show plot
plt.show()



















# -------------- PART 5: RECORD STATS --------------

# Make a text file
output_file = f"new_paramspace_sensfuncs_{choice1}_and_{choice2}.txt"
with open(output_file, "w") as file:

    # Record system info
    file.write("------------\n")
    file.write("System info:\n")
    file.write("------------\n")
    file.write(f"Sensitivity function 1: {label1}\n")
    file.write(f"Sensitivity function 2: {label2}\n")

    # Record MOSA stats
    file.write("--------------------------------------------\n")
    file.write("Initial sampling to probe system properties:\n")
    file.write("--------------------------------------------\n")
    file.write(f"alpha values from {alpha_min} to {alpha_max} with {alpha_sampsize} linspaced samples\n")
    file.write(f"n values from {n_min} to {n_max} with {n_sampsize} linspaced samples\n")
    file.write(f"Sampled energy difference in S_alpha sensitivity function: {deltaE_sens1}\n")
    file.write(f"Sampled energy difference in S_n sensitivity function: {deltaE_sens2}\n")
    file.write(f"Sampled cumulative energy difference: {deltaE}\n")
    file.write(f"Probability of transitioning to a higher energy state in hot run: {probability_hot}\n")
    file.write(f"Hot run tempertaure: {temp_hot}\n")
    file.write(f"Probability of transitioning to a higher energy state in cold run: {probability_cold}\n")
    file.write(f"Cold run tempertaure: {temp_cold}\n")

    # Record bounds
    file.write("-------------------------------------------------\n")
    file.write(f"Bounds of new parameter space after {runs} runs:\n")
    file.write("-------------------------------------------------\n")
    file.write(f"alpha_min: {min_vals[0]}, alpha_max: {max_vals[0]}\n")
    file.write(f"n_min: {min_vals[1]}, n_max: {max_vals[1]}\n")
    file.write(f"New parameter space is {percentage:.2f}% of original parameter space volume.")
    
    # Record reduced parameter space stats
    file.write("-------------------------------------------------\n")
    file.write(f"Stats of new parameter space after {runs} runs:\n")
    file.write("-------------------------------------------------\n")
    file.write(f"Density: {(a_density * n_density) / ((alpha_max-alpha_min) * (n_max-n_min))} points per unit area \n")
    file.write(f"Number of points: {np.shape(inside_rect_points)[0]}\n")