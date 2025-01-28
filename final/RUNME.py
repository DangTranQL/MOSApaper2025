# -------------- PART 0: PYTHON PRELIM --------------

import importlib
import os
import time
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
import pyvista as pv

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from math import inf
from numpy.random import seed
from scipy.spatial import ConvexHull, Delaunay

# -------------- PART 0: CHOOSE CIRCUIT AND SET UP FOLDER --------------

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

# dy/dt
Equ2 = config.Equ2
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    y = P[1]
    beta_x = params[0]
    beta_y = params[1]
    n      = params[2]
    val0 = Equ1(x, y, beta_x, n)
    val1 = Equ2(x, y, beta_y, n)
    return np.array([val0, val1])

# Define values
t = 0.0

# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------

# Define analytical sensitivity expressions
S_betax_xss_analytic = config.S_betax_xss_analytic
S_betax_yss_analytic = config.S_betay_xss_analytic
S_betay_xss_analytic = config.S_betay_xss_analytic
S_betay_yss_analytic = config.S_betay_yss_analytic
S_n_xss_analytic = config.S_n_xss_analytic
S_n_yss_analytic = config.S_n_yss_analytic

# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------

# Print prompt
print("""
We have the following sensitivity functions:
0. |S_betax_xss|
1. |S_betax_yss|
2. |S_betay_xss|
3. |S_betay_yss|
4. |S_n_xss|
5. |S_n_yss|
""")

# Choose pair of functions
choice1 = int(input("Please select first option number:"))
choice2 = int(input("Please select second option number:"))

# List of sensitivity function names
sensitivity_labels = [
    "|S_betax_xss|",
    "|S_betax_yss|",
    "|S_betay_xss|",
    "|S_betay_yss|",
    "|S_n_xss|",
    "|S_n_yss|"]

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

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS AND YSS GIVEN SOME INITIAL GUESS
def ssfinder(beta_x_val,beta_y_val,n_val):

    # Define initial guesses
    InitGuesses = config.generate_initial_guesses(beta_x_val, beta_y_val)
    
    # Define array of parameters
    params = np.array([beta_x_val, beta_y_val, n_val])
    
    # No valid solution initially
    found_valid = False
    
    # For each until you get one that gives a solution or you exhaust the list
    for InitGuess in InitGuesses:

        # Get solution details
        output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)
        xss, yss = output
        fvec = infodict['fvec'] 

        # Check if stable attractor point
        delta = 1e-8
        dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta
        dEqudy = (Equs([xss,yss+delta], t, params)-Equs([xss,yss], t, params))/delta
        jac = np.transpose(np.vstack((dEqudx,dEqudy)))
        eig = np.linalg.eig(jac)[0]
        instablility = np.any(np.real(eig) >= 0)

        # Check if valid solution
        if xss > 0.04 and yss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:
            found_valid = True
            return xss, yss 
            # Note: 
            # xss and yss are both of type numpy.float64
            # Eg: 
            # ssfinder(2,2,2) == (1.0000000108275326, 1.0000000108275326)
            # type(ssfinder(2,2,2)) == tuple
            # type(ssfinder(2,2,2)[0]) == type(ssfinder(2,2,2)[1]) == numpy.float64

    # If not valid, set to nan
    if not found_valid:
        return float('nan'), float('nan')

# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(xss_list, yss_list, n_list, beta_x_list, beta_y_list, choice1, choice2):
    
    # Evaluate sensitivities
    S_betax_xss = S_betax_xss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)
    S_betax_yss = S_betax_yss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)
    S_betay_xss = S_betay_xss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)
    S_betay_yss = S_betay_yss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)
    S_n_xss     =     S_n_xss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)
    S_n_yss     =     S_n_yss_analytic(xss_list, yss_list, n_list, beta_x_list, beta_y_list)

    # Sensitivity dictionary
    sensitivities = {
        "S_betax_xss": S_betax_xss,
        "S_betax_yss": S_betax_yss,
        "S_betay_xss": S_betay_xss,
        "S_betay_yss": S_betay_yss,
        "S_n_xss": S_n_xss,
        "S_n_yss": S_n_yss}

    # Map indices to keys
    labels = {
        0: "S_betax_xss",
        1: "S_betax_yss",
        2: "S_betay_xss",
        3: "S_betay_yss",
        4: "S_n_xss",
        5: "S_n_yss"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]
    # Note: 
    # Both outputs are of type numpy.ndarray
    # Eg:
    # xss_list = np.array((1.0000000108275326, 2.8793852415718164))
    # yss_list = np.array((2.8793852415718164, 1.0000000108275326))
    # n_list = np.array([2,3])
    # beta_x_list = np.array([2,3])
    # beta_y_list = np.array([2,3])
    # choice1 = 0
    # choice2 = 1
    # ans = senpair(xss_list, yss_list, n_list, beta_x_list, beta_y_list, choice1, choice2) == (array([4.61785728e+07, 1.01476269e+00]), array([ 0.53265904, 24.00004561]))
    # type(ans) = tuple
    # type(ans[0]) == type(ans[1]) == numpy.ndarray
    
# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):
	
	# Update parameter set
    beta_x_val = solution["beta_x"]
    beta_y_val = solution["beta_y"]
    n_val = solution["n"]

    # Create an empty numpy array
    xss_collect = np.array([])
    yss_collect = np.array([])

    # Find steady states and store
    xss, yss = ssfinder(beta_x_val,beta_y_val,n_val)
    xss_collect = np.append(xss_collect,xss)
    yss_collect = np.append(yss_collect,yss)
    
    # Get sensitivity pair
    sens1, sens2 = senpair(xss_collect, yss_collect, solution["n"], solution["beta_x"], solution["beta_y"], choice1, choice2)
    ans1 = float(sens1) # what if ans1 has more than one element?
    ans2 = float(sens2) # what if ans2 has more than one element?
    return ans1, ans2
    

# -------------- PART 1: GAUGING MOSA PARAMETERS --------------

# Sample beta_x values
beta_x_min = float(input("Please enter minimum beta_x value: "))
beta_x_max = float(input("Please enter maximum beta_x value: "))
beta_x_sampsize = int(input("Please enter the number of beta_x samples: "))
beta_x_samps = np.linspace(beta_x_min, beta_x_max, beta_x_sampsize)

# Sample beta_y values
beta_y_min = float(input("Please enter minimum beta_y value: "))
beta_y_max = float(input("Please enter maximum beta_y value: "))
beta_y_sampsize = int(input("Please enter the number of beta_y samples: "))
beta_y_samps = np.linspace(beta_y_min, beta_y_max, beta_y_sampsize)

# Sample n values
n_min = float(input("Please enter minimum n value: "))
n_max = float(input("Please enter maximum n value: "))
n_sampsize = int(input("Please enter the number of n samples: "))
n_samps = np.linspace(n_min, n_max, n_sampsize)

# Create empty arrays to store corresponding values of xss and yss
xss_samps = np.array([])
yss_samps = np.array([])
sens1_samps = np.array([])
sens2_samps = np.array([])

# For each combination of parameters
for i in beta_x_samps:
    for j in beta_x_samps:
        for k in n_samps:
            # Get steady states and store
            xss, yss = ssfinder(i,j,k)
            xss_samps = np.append(xss_samps,xss)
            yss_samps = np.append(yss_samps,yss)
            # Get sensitivities and store
            sens1, sens2 = senpair(xss, yss, k, i, j, choice1, choice2)
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

# -------------- PART 2: PREPPING MOSA --------------

# Print prompt
print("Now ready to MOSA...")

# Get number of MOSA runs
runs = int(input("Number of MOSA runs you would like to complete (if in doubt enter 10): "))
print("MOSA runs = ", runs)

# Define lists to collect Pareto-optimal parameter values from each MOSA run
pareto_sensfunc1_collection = []
pareto_sensfunc2_collection = []
pareto_betax_collection     = []
pareto_betay_collection     = []
pareto_n_collection         = []

# Define function to ensure parameters of MOSA solutions are positive
def validate_solution(solution):
    solution["beta_x"] = max(solution["beta_x"], 0.01)
    solution["beta_y"] = max(solution["beta_y"], 0.01)
    solution["n"]      = max(solution["n"]     , 0.01)
    return 

# For each run
for run in range(runs):
    print(f"MOSA run number: {run}")

	# -------------- PART 2a: ANNEAL TO GET PARETO FRONT IN SENSITIVITY SPACE --------------
	
	# Set random number generator to 0
    seed(0)
	
	# Initialisation of MOSA
    opt = mosa.Anneal()
    opt.archive_size = 100
    opt.maximum_archive_rejections = opt.archive_size
    opt.population = {"beta_x": (beta_x_min, beta_x_max), "beta_y": (beta_y_min, beta_y_max), "n": (n_min, n_max)}
	
	# Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = 100
    no_of_steps_from_hot_to_cold = int(np.ceil((temp_hot-temp_cold)/opt.temperature_decrease_factor))
    opt.number_of_temperatures = no_of_steps_from_hot_to_cold
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"beta_x":1, "beta_y":1, "n":1}
    step_scaling = 0.2
    opt.mc_step_size= {"beta_x": (beta_x_max-beta_x_min)*step_scaling , "beta_y": (beta_y_max-beta_y_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Hot run
    start_time = time.time()
    # opt.evolve(lambda sol: fobj(validate_solution(sol)))
    opt.evolve(fobj)
    print(f"Hot run time: {time.time() - start_time} seconds")
	
    # Cold run options
    opt.initial_temperature = temp_cold
    opt.number_of_iterations = 100
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"alpha":1,"n":1}
    step_scaling = 0.03
    opt.mc_step_size= {"beta_x": (beta_x_max-beta_x_min)*step_scaling , "beta_y": (beta_y_max-beta_y_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Cold run
    start_time = time.time()
    # opt.evolve(lambda sol: fobj(validate_solution(sol)))
    opt.evolve(fobj)
    print(f"Cold run time: {time.time() - start_time} seconds")
    
    # Output 
    start_time = time.time()
    pruned = opt.prunedominated()
    opt.plotfront(pruned)
    print(f"Pruning time: {time.time() - start_time} seconds")
	
	# -------------- PART 2b: PLOT NON PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["beta_x"] for solution in data["Solution"]])
    print(f"Archive length: {length}")
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Unpruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'unpruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
    # -------------- PART 2c: SAVE POINTS --------------
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_sensfunc1_collection.append(dummy1)
        pareto_sensfunc2_collection.append(dummy2)
	
    # -------------- PART 2d: CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract alpha and n values from the solutions
    beta_x_values = [solution["beta_x"] for solution in data["Solution"]]
    beta_y_values = [solution["beta_y"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(beta_x_values, beta_y_values, n_values)
    ax.set_xlabel('beta_x')
    ax.set_ylabel('beta_y')
    ax.set_zlabel('n')
    ax.set_title(f'Unpruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'unpruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()
    
    # -------------- PART 2e: SAVE POINTS --------------
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3 in zip(beta_x_values, beta_y_values, n_values):
        pareto_betax_collection.append(dummy1)
        pareto_betay_collection.append(dummy2)
        pareto_n_collection.append(dummy3)
    
#    # -------------- PART 2f: DELETE ARCHIVE AND CHECKPOINT FILES --------------
#
#    # File names
#    files_to_delete = ["archive.json", "checkpoint.json"]
#    
#    # For each file name
#    for file_name in files_to_delete:
#        # Delete if exists
#        if os.path.exists(file_name):
#            os.remove(file_name)
#            print(f"{file_name} has been deleted.")
#        # Prompt that it does not exist otherwise
#        else:
#            print(f"{file_name} does not exist.")

# -------------- PART 3: CUMULATIVE NEW PARAMETER SPACE --------------

# 3a: FIND RECTANGULAR PRISM BOUNDS

# Define scattered points in 3D space
points = np.array(list(zip(pareto_betax_collection, pareto_betay_collection, pareto_n_collection)))
# Define rectangular prism that bounds the scatter
min_vals = np.min(points, axis=0)
max_vals = np.max(points, axis=0)

# 3b: COMPARE OLD VS NEW PARAM SPACE VOLUMES

# Volume of the bounding rectangular prism
new_param_vol = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1]) * (max_vals[2] - min_vals[2])
# Volume of original parameter space
old_param_vol = (beta_x_max-beta_x_min) * (beta_y_max-beta_y_min) * (n_max-n_min)
# Polyhedron's volume as percentage of parameter space
percentage = (new_param_vol / old_param_vol) * 100
# Print percentage
print(f"New parameter space is {percentage:.2f}% of original parameter space volume.")

# 3c: PLOT OLD VS PARAM SPACE

# Create the bounding box of the new parameter space
new_param_box = pv.Box(bounds=(min_vals[0], max_vals[0], min_vals[1], max_vals[1], min_vals[2], max_vals[2]))
# Create the original parameter space box
old_param_box = pv.Box(bounds=(beta_x_min, beta_x_max, beta_y_min, beta_y_max, n_min, n_max))
# Create a plotter
plotter = pv.Plotter()
# Show each param space with different colors
plotter.add_mesh(new_param_box, color='blue', opacity=0.3, label='New Parameter Space')
plotter.add_mesh(old_param_box, color='red', opacity=0.3, label='Original Parameter Space')
# Add legend
plotter.add_legend()
# Add a bounding box with axes labels and ticks
plotter.show_bounds(
    grid="front",          # Display bounds on the front grid
    location="outer",      # Place axes outside the bounding box
    ticks="both",          # Show ticks on all axes
    xlabel="X Axis",       # Label for X axis
    ylabel="Y Axis",       # Label for Y axis
    zlabel="Z Axis",       # Label for Z axis
)
# Show plot
plotter.show()
# Save plot
output_file = f"paramspaces_sensfuncs_{choice1}_and_{choice2}.png"
plotter.screenshot(output_file)



# ------------------------------------------------------------------------------------------------------------------

# If we have enough points to make a convex hull (at least 4 points)
if len(points)>3:
    
    # Step 2: Compute the convex hull
    hull = ConvexHull(points)
    
    # Step 3: Create a Delaunay triangulation for point inclusion
    tri = Delaunay(points[hull.vertices])
    
    # Step 4: Define the voxel grid
    # Define grid spacing (distance between points)
    grid_spacing = 0.05
    # Define bounding box
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    # Create a 3D grid of points
    x = np.arange(min_bounds[0], max_bounds[0], grid_spacing)
    y = np.arange(min_bounds[1], max_bounds[1], grid_spacing)
    z = np.arange(min_bounds[2], max_bounds[2], grid_spacing)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    
    # Step 5: Check inclusion in the convex hull
    mask = tri.find_simplex(points) >= 0
    hull_samples = grid_points[mask]
    
    # Step 6: Plot the convex hull
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", alpha=0.5, label="Original Points")
    for simplex in hull.simplices:
        ax1.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], "k-")
    ax1.set_title("Convex Hull")
    plt.show()
    
    # Step 7: Plot the sampled points inside the convex hull
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.scatter(hull_samples[:, 0], hull_samples[:, 1], evenly_spaced_points[:, 2], c="red", alpha=0.8, label="Sampled Points")
    ax2.set_title("Sampled Points Inside Convex Hull")
    plt.show()
    
    # Output the result
    print(f"Number of evenly spaced points: {len(evenly_spaced_points)}")
# ------------------------------------------------------------------------------------------------------------------


# -------------- PART 4: RECORD STATS --------------

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
    file.write(f"beta_x values from {beta_x_min} to {beta_x_max} with {beta_x_sampsize} linspaced samples\n")
    file.write(f"beta_y values from {beta_y_min} to {beta_y_max} with {beta_y_sampsize} linspaced samples\n")
    file.write(f"n values from {n_min} to {n_max} with {n_sampsize} linspaced samples\n")
    file.write(f"Sampled energy difference in 1st sensitivity function: {deltaE_sens1}\n")
    file.write(f"Sampled energy difference in 2nd sensitivity function: {deltaE_sens2}\n")
    file.write(f"Sampled cumulative energy difference: {deltaE}\n")
    file.write(f"Probability of transitioning to a higher energy state in hot run: {probability_hot}\n")
    file.write(f"Hot run tempertaure: {temp_hot}\n")
    file.write(f"Probability of transitioning to a higher energy state in cold run: {probability_cold}\n")
    file.write(f"Cold run tempertaure: {temp_cold}\n")

    # Record bounds
    file.write("-------------------------------------------------\n")
    file.write(f"Bounds of new parameter space after {runs} runs:\n")
    file.write("-------------------------------------------------\n")
    file.write(f"beta_x_min: {min_vals[0]}, beta_x_max: {max_vals[0]}\n")
    file.write(f"beta_y_min: {min_vals[1]}, beta_y_max: {max_vals[1]}\n")
    file.write(f"n_min: {min_vals[2]}, n_max: {max_vals[2]}\n")
    file.write(f"New parameter space is {percentage:.2f}% of original parameter space volume.")
    
# -------------- PART 5: PERFORM SEARCH --------------
