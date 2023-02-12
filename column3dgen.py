import numpy as np
import mpm_3dinput_utils as inputgen
import box_ranges_gen
# %matplotlib qt

# inputs
trajectory_names = []
for i in range(0, 3):
    trajectory_names.append(f"3dsand_test{i}")
num_particle_groups = 2
cellsize = 0.1
outer_cell_thickness = cellsize/8
simulation_domain = [[0.0, 1.0 + outer_cell_thickness*2],
                     [0.0, 1.0 + outer_cell_thickness*2],
                     [0.0, 1.0 + outer_cell_thickness*2]]  # simulation domain. Particle group are generated inside this domain
particle_domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.7]]  # limit where the particle groups are generated.
particle_length = [0.25, 0.25, 0.25]  # dimension of particle group
vel_bound = [-2, 2]  # lower and upper limits for random velocity vector for a particle group



trajectory_info = {}
    # trajectory_info[f"{trajectory_name}"] = {}
for trajectory_name in trajectory_names:
    trajectory_info[f"{trajectory_name}"] = {
        'domain': simulation_domain,
        'cellsize': cellsize,
        'outer_cell_thickness': outer_cell_thickness,
        'particle_info': {}
    }
    # box_ranges = box_ranges_gen.make_n_box_ranges(
    #     num_particle_groups=num_particle_groups,
    #     size=particle_length,
    #     domain=particle_domain,
    #     size_random_level=0.15,
    #     boundary_offset=[cellsize/2, cellsize/2, cellsize/2],
    #     min_interval=cellsize/8,
    #     dimensions=3
    # )
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
                           outer_cell_thickness=sim_info["outer_cell_thickness"],
                           particle_info=sim_info["particle_info"]
                           )