import numpy as np
import mpm_input_utils as inputgen
# %matplotlib qt

# inputs
trajectory_names = []
for i in range(0, 5):
    trajectory_names.append(f"3dsand_test{i}")
num_particle_groups = 1
simulation_domain = [[0.0, 1.0], [0.0, 1.0]]  # simulation domain. Particle group are generated inside this domain
particle_domain = [[0.0, 1.0], [0.0, 1.0]]  # limit where the particle groups are generated.
particle_length = [0.30, 0.30]  # dimension of particle group
vel_bound = [-2, 2]  # lower and upper limits for random velocity vector for a particle group
cellsize = 0.08


trajectory_info = {}
    # trajectory_info[f"{trajectory_name}"] = {}
for trajectory_name in trajectory_names:
    trajectory_info[f"{trajectory_name}"] = {
        'domain': simulation_domain,
        'particle_info': {}
    }
    particle_ranges = inputgen.particle_ranges(
        num_particle_groups=num_particle_groups,
        domain=particle_domain,
        particle_length=particle_length,
        boundary_offset=cellsize,
        dims=3
    )
    for i in range(num_particle_groups):
        init_vel = [vel for vel in np.random.uniform(vel_bound[0], vel_bound[1], 3)]

        trajectory_info[f"{trajectory_name}"]['particle_info'][f"group{i+1}"] = {
            "bound": particle_ranges[i],
            "init_vel": init_vel
        }

for trj_name, sim_info in trajectory_info.items():
    inputgen.mpm_input_gen(save_name=trj_name,
                           domain=sim_info["domain"],
                           cell_size=cellsize,
                           particle_info=sim_info["particle_info"]
                           )