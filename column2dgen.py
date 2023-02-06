import numpy as np
from mpm_2dinput_utils import Column2DSimulation
from mpm_2dinput_utils import make_n_box_ranges
import os
import json
import argparse
from matplotlib import pyplot as plt
import sys
from absl import app

def main(_):
    random_gen = True
    save_path = "mpm_inputs"
    trajectory_names = ["sand3d-0", "sand3d-1"]
    simulation_domain = [[0.0, 1.0], [0.0, 1.0]]
    cellsize = 0.025
    nparticle_perdim_percell = 4
    particle_randomness = 0.8
    wall_friction = 0.27

    num_particle_groups = 3
    # define materials
    material0 = {
        "id": 0,
        "type": "MohrCoulomb3D",
        "density": 1800,
        "youngs_modulus": 2000000.0,
        "poisson_ratio": 0.3,
        "friction": 30,
        "dilation": 0.0,
        "cohesion": 100,
        "tension_cutoff": 50,
        "softening": False,
        "peak_pdstrain": 0.0,
        "residual_friction": 30,
        "residual_dilation": 0.0,
        "residual_cohesion": 0.0,
        "residual_pdstrain": 0.0
    }
    material_id = [0, 0, 0]  # material id of each particle group
    # (maybe make a dict named `trajectory_info` with `simulation_domain`)
    if random_gen is True:
        particle_length = [0.2, 0.2]  # length of cube for x, y dir
        particle_gen_candidate_area = [[0.0, 1.0], [0.0, 0.7]]
        range_randomness = 0.2
        vel_bound = [-2.0, 2.0]
    else:  # type particle group info
        pass
        # particle_domain = [[]]


    # simulation options
    analysis = {
        "type": "MPMExplicit2D",
        "mpm_scheme": "usf",
        "locate_particles": False,
        "dt": 1e-05,
        "damping": {
            "type": "Cundall",
            "damping_factor": 0.05
        },
        "velocity_update": False,
        "nsteps": 105000,
        "uuid": "3dsand_test0"
    }
    post_processing = {
        "path": "results/",
        "output_steps": 250,
        "vtk": ["stresses", "displacements"]
    }

    # Create trajectory meta data
    metadata = {}
    for i, trajectory_name in enumerate(trajectory_names):

        # general simulation config
        metadata[f"simulation{i}"] = {}
        metadata[f"simulation{i}"]["name"] = trajectory_name
        metadata[f"simulation{i}"]["simulation_domain"] = simulation_domain
        metadata[f"simulation{i}"]["cellsize"] = cellsize
        metadata[f"simulation{i}"]["nparticle_perdim_percell"] = nparticle_perdim_percell
        metadata[f"simulation{i}"]["particle_randomness"] = particle_randomness

        # particle config for each simulation
        metadata[f"simulation{i}"]["particle"] = {}
        # random gen for ranges where particles are generated
        if random_gen is True:
            particle_ranges = make_n_box_ranges(num_particle_groups=num_particle_groups,
                      size=particle_length,
                      domain=particle_gen_candidate_area,
                      boundary_offset=cellsize,
                      size_random_level=range_randomness)

            # assign the generated ranges, vel, and materials for each particle group
            for g in range(num_particle_groups):
                metadata[f"simulation{i}"]["particle"][f"group{g}"] = {
                    "particle_domain": particle_ranges[g],
                    "material_id": material_id[g],
                    "particle_vel":  [vel for vel in np.random.uniform(vel_bound[0], vel_bound[1], 2)]
                }



    # init
    sim = Column2DSimulation(simulation_domain=simulation_domain,
                             cellsize=cellsize,
                             npart_perdim_percell=nparticle_perdim_percell,
                             randomness=particle_randomness,
                             wall_friction=wall_friction,
                             analysis=analysis,
                             post_processing=post_processing)

    # gen mpm inputs
    for simulation in metadata.values():

        # write mesh
        mesh_info = sim.create_mesh()
        sim.write_mesh_file(mesh_info, save_path=f"{save_path}/{simulation['name']}")

        # write particle
        particle_info = sim.create_particle(simulation["particle"])
        sim.write_particle_file(particle_info, save_path=f"{save_path}/{simulation['name']}")

        # write entity
        sim.write_entity(save_path=f"{save_path}/{simulation['name']}",
                         mesh_info=mesh_info,
                         particle_info=particle_info)

        # write mpm.json
        sim.mpm_inputfile_gen(
            save_path=f"{save_path}/{simulation['name']}",
            material_types=[material0],
            particle_info=particle_info)

if __name__ == '__main__':
    app.run(main)
