import json

import numpy as np
import random
from absl import app
import os
import box_ranges_gen
from mpm_2dinput_utils import Column2DSimulation


def main(_):

     ######## Inputs #######
    dims = 3
    random_gen = True
    save_path = "mpm_inputs"
    trajectory_names = ["sand3d-0"]
    cellsize = 0.1
    outer_cell_thickness = cellsize / 4
    simulation_domain = [
        [0.0, 1.0+outer_cell_thickness*2],
        [0.0, 1.0+outer_cell_thickness*2],
        [0.0, 1.0+outer_cell_thickness*2]]
    nparticle_perdim_percell = 4
    particle_randomness = 0.8
    wall_friction = 0.27

    num_particle_groups = 2
    # define materials
    materials = [
        {
            "id": 0,
            "type": "MohrCoulomb3D" if dims == 3 else "MohrCoulomb2D",
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
        },
        {
            "id": 2,
            "type": "MohrCoulomb3D" if dims == 3 else "MohrCoulomb2D",
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
    ]
    material_id = [0, 0]  # material id of each particle group
    if len(material_id) is not num_particle_groups:
        raise Exception("`num_particle_groups` should match len(material_id)")
    if random_gen is True:
        particle_length = [0.20, 0.20, 0.20]  # length of cube for x, y dir
        particle_gen_candidate_area = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.7]]
        range_randomness = 0.2
        vel_bound = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 1.0]]
    else:  # type particle group info
        pass
        # particle_domain = [[]]

    # simulation options
    analysis = {
        "type": "MPMExplicit3D" if dims == 3 else "MPMExplicit2D",
        "mpm_scheme": "usf",
        "locate_particles": False,
        "dt": 1e-05,
        "damping": {
            "type": "Cundall",
            "damping_factor": 0.05
        },
        "resume": {
            "resume": False,
            "uuid": "3dsand" if dims == 3 else "2dsand",
            "step": 0
        },
        "velocity_update": False,
        "nsteps": 105000,
        "uuid": "3dsand" if dims == 3 else "2dsand",

    }
    post_processing = {
        "path": "results/",
        "output_steps": 250,
        "vtk": ["stresses", "displacements"]
    }
     ###############

    # Create trajectory meta data
    metadata = {}
    for i, trajectory_name in enumerate(trajectory_names):

        # general simulation config
        metadata[f"simulation{i}"] = {}
        metadata[f"simulation{i}"]["name"] = trajectory_name
        metadata[f"simulation{i}"]["cellsize"] = cellsize
        metadata[f"simulation{i}"]["outer_cell_thickness"] = outer_cell_thickness
        metadata[f"simulation{i}"]["simulation_domain"] = simulation_domain
        metadata[f"simulation{i}"]["nparticle_perdim_percell"] = nparticle_perdim_percell
        metadata[f"simulation{i}"]["particle_randomness"] = particle_randomness

        # particle config for each simulation
        metadata[f"simulation{i}"]["particle"] = {}
        # random gen for ranges where particles are generated
        particle_ranges = box_ranges_gen.make_n_box_ranges(
            num_particle_groups=num_particle_groups,
            size=particle_length,
            domain=particle_gen_candidate_area,
            size_random_level=range_randomness,
            boundary_offset=[outer_cell_thickness for _ in range(dims)],
            min_interval=cellsize/2,
            dimensions=dims
        )

        # assign the generated ranges, vel, and materials for each particle group
        for g in range(num_particle_groups):
            metadata[f"simulation{i}"]["particle"][f"group{g}"] = {
                "particle_domain": particle_ranges[g],
                "particle_vel":  [random.uniform(vel_bound[i][0], vel_bound[i][1]) for i in range(dims)],
                "material_id": material_id[g]
            }

        if not os.path.exists(f"{save_path}/{trajectory_name}"):
            os.makedirs(f"{save_path}/{trajectory_name}")
        out_file = open(f"{save_path}/{trajectory_name}/metadata.json", "w")
        json.dump(metadata[f"simulation{i}"], out_file, indent=2)
        out_file.close()


    # init
    sim = Column2DSimulation(simulation_domain=simulation_domain,
                             cellsize=cellsize,
                             outer_cell_thickness=outer_cell_thickness,
                             npart_perdim_percell=nparticle_perdim_percell,
                             randomness=particle_randomness,
                             wall_friction=wall_friction,
                             post_processing=post_processing,
                             dims=dims)

    # gen mpm inputs
    for simulation in metadata.values():

        # write mesh
        mesh_info = sim.create_mesh()
        sim.write_mesh_file(mesh_info, save_path=f"{save_path}/{simulation['name']}")

        # write particle
        particle_info = sim.create_particle(simulation["particle"])
        sim.write_particle_file(particle_info, save_path=f"{save_path}/{simulation['name']}")
        sim.plot_particle_config(particle_group_info=particle_info, save_path=f"{save_path}/{simulation['name']}")

        # write entity
        sim.write_entity(save_path=f"{save_path}/{simulation['name']}",
                         mesh_info=mesh_info,
                         particle_info=particle_info)

        # write mpm.json
        sim.mpm_inputfile_gen(
            save_path=f"{save_path}/{simulation['name']}",
            material_types=materials,
            particle_info=particle_info,
            analysis=analysis)
        # write mpm.json
        sim.mpm_inputfile_gen(
            save_path=f"{save_path}/{simulation['name']}",
            material_types=materials,
            particle_info=particle_info,
            analysis=analysis,
            resume=True)

if __name__ == '__main__':
    app.run(main)
