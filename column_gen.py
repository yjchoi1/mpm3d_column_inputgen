import json

import numpy as np
import random
from absl import app
import os
import box_ranges_gen
from mpm_2dinput_utils import ColumnSimulation

ndims = 3
save_path = "/work2/08264/baagee/frontera/gns-mpm-data/mpm/sand3d/"
random_gen = False
density = 1800  # assume all material has the same density
simulation_case = "sand3d_column_collapse"

if random_gen == False:
    input_metadata_name = "metadata-sand3d_column_collapse15"
    data_tag = [str(tag) for tag in range(15, 16)]
    k0 = None

elif random_gen == True:
    k0 = None  # for computing geostatic stress for particles. Set `None` if not considering it
    data_tag = [str(tag) for tag in range(0, 4)]
    trajectory_names = [f"{simulation_case}{i}" for i in data_tag]
    ncells_per_dim = [12, 3, 12]  ####
    outer_cell_thickness = 1.0/12/4
    simulation_domain = [[0.0, 1.0],
                         [-0.125, 0.125],
                         [0.0, 1.0]]
    if len(simulation_domain) is not ndims:
        raise Exception("`simulation_domain` should match `ndims`")
    nparticle_perdim_percell = 4
    particle_randomness = 0.7
    wall_friction = 0.385
    num_particle_groups = 1
    material_id = [0]  # material id associated with each particle group
    if len(material_id) is not num_particle_groups:
        raise Exception("`num_particle_groups` should match len(material_id)")

    # About particles
    particle_length = [0.3, 0.1, 0.3]  # length of cube for x, y dir
    particle_gen_candidate_area = [[0.0, 1.0], [0.0, 0.25], [0.0, 1.0]]
    range_randomness = 0.0
    vel_bound = [[0, 0], [0, 0], [0, 0]]
    # error
    if len(particle_length) != ndims or len(particle_gen_candidate_area) != ndims or len(vel_bound) != ndims:
        raise Exception("particle related inputs should match `ndims`")

else:
    raise Exception("Either `random_gen` should be either true or false")

# define materials
material_model = "MohrCoulomb3D" if ndims == 3 else "MohrCoulomb2D"
materials = [
    {
        "id": 0,
        "type": material_model,
        "density": density,
        "youngs_modulus": 2000000.0,
        "poisson_ratio": 0.3,
        "friction": 30.0,
        "dilation": 0.0,
        "cohesion": 100,
        "tension_cutoff": 50,
        "softening": False,
        "peak_pdstrain": 0.0,
        "residual_friction": 30.0,
        "residual_dilation": 0.0,
        "residual_cohesion": 0.0,
        "residual_pdstrain": 0.0
    }
]

# simulation options
analysis = {
    "type": "MPMExplicit3D" if ndims == 3 else "MPMExplicit2D",
    "mpm_scheme": "usf",
    "locate_particles": False,
    "dt": 1e-06,
    "damping": {
        "type": "Cundall",
        "damping_factor": 0.05
    },
    "resume": {
        "resume": False,
        "uuid": f"{simulation_case}",
        "step": 0
    },
    "velocity_update": False,
    "nsteps": 950000,
    "uuid": f"{simulation_case}"
}

analysis_resume = {
    "type": "MPMExplicit3D" if ndims == 3 else "MPMExplicit2D",
    "mpm_scheme": "usf",
    "locate_particles": False,
    "dt": 1e-06,
    "damping": {
        "type": "Cundall",
        "damping_factor": 0.05
    },
    "resume": {
        "resume": True,
        "uuid": f"{simulation_case}",
        "step": 0
    },
    "velocity_update": False,
    "nsteps": 950000,
    "uuid": f"{simulation_case}"
}

post_processing = {
    "path": "results/",
    "output_steps": 2500,
    "vtk": ["displacements"]
}

def main(_):
    # Create trajectory meta data
    metadata = {}
    if random_gen == True:
        for i, trajectory_name in enumerate(trajectory_names):

            # general simulation config
            metadata[f"simulation{i}"] = {}
            metadata[f"simulation{i}"]["name"] = trajectory_name
            metadata[f"simulation{i}"]["ncells_per_dim"] = ncells_per_dim
            metadata[f"simulation{i}"]["outer_cell_thickness"] = outer_cell_thickness
            metadata[f"simulation{i}"]["simulation_domain"] = simulation_domain
            metadata[f"simulation{i}"]["nparticle_perdim_percell"] = nparticle_perdim_percell
            metadata[f"simulation{i}"]["particle_randomness"] = particle_randomness
            metadata[f"simulation{i}"]["k0"] = k0
            metadata[f"simulation{i}"]["wall_friction"] = wall_friction

            # compute cellsize_per_dim and assume it is the same for all dims
            cellsize_per_dim = [
                (metadata[f"simulation{i}"]["simulation_domain"][dim][1] -
                metadata[f"simulation{i}"]["simulation_domain"][dim][0]) /
                metadata[f"simulation{i}"]["ncells_per_dim"][dim] for dim in range(ndims)
            ]
            if not all(cellsize == cellsize_per_dim[0] for cellsize in cellsize_per_dim):
                raise NotImplementedError("All cell size per dim should be the same")

            # particle config for each simulation
            metadata[f"simulation{i}"]["particle"] = {}
            # random gen for ranges where particles are generated
            particle_ranges = box_ranges_gen.make_n_box_ranges(
                num_particle_groups=num_particle_groups,
                size=particle_length,
                domain=particle_gen_candidate_area,
                size_random_level=range_randomness,
                min_interval=cellsize_per_dim[0]
            )

            # assign the generated ranges, vel, and materials for each particle group
            for g in range(num_particle_groups):
                metadata[f"simulation{i}"]["particle"][f"group{g}"] = {
                    "particle_domain": particle_ranges[g],
                    "particle_vel": [random.uniform(vel_bound[i][0], vel_bound[i][1]) for i in range(ndims)],
                    "material_id": material_id[g]
                }

            # save metadata individually for each sim folder
            if not os.path.exists(f"{save_path}/{trajectory_name}"):
                os.makedirs(f"{save_path}/{trajectory_name}")
            out_file = open(f"{save_path}/{trajectory_name}/metadata.json", "w")
            json.dump(metadata[f"simulation{i}"], out_file, indent=2)
            out_file.close()
        # save metadata for entire sims
        out_file = open(f"{save_path}/metadata-{simulation_case}.json", "w")
        json.dump(metadata, out_file, indent=2)
        out_file.close()

    elif random_gen == False:
        # read metadata
        f = open(f"{save_path}/{input_metadata_name}.json")
        metadata = json.load(f)

        # make simulation name which is the name of the folder to save results
        user_specified_trajectory_names = [f"{simulation_case}{i}" for i in data_tag]

        # save metadata individually for each sim folder
        for i, (sim, siminfo) in enumerate(metadata.items()):
            if not os.path.exists(f"{save_path}/{user_specified_trajectory_names[i]}"):
                os.makedirs(f"{save_path}/{user_specified_trajectory_names[i]}")
            # change simulation name (which is the folder name to save results) to user specified name.
            siminfo["name"] = user_specified_trajectory_names[i]
            # save each simulation information to each specified result folder
            out_file = open(f"{save_path}/{siminfo['name']}/metadata.json", "w")
            json.dump(metadata[sim], out_file, indent=2)
            out_file.close()
        # save entire metadata for all simulation in save_path
        out_file = open(f"{save_path}/{input_metadata_name}.json", "w")
        json.dump(metadata, out_file, indent=2)

    # init
    sim = ColumnSimulation(simulation_domain=metadata["simulation0"]["simulation_domain"],
                           ncells_per_dim=metadata["simulation0"]["ncells_per_dim"],
                           outer_cell_thickness=metadata["simulation0"]["outer_cell_thickness"],
                           npart_perdim_percell=metadata["simulation0"]["nparticle_perdim_percell"],
                           randomness=metadata["simulation0"]["particle_randomness"],
                           wall_friction=metadata["simulation0"]["wall_friction"],
                           post_processing=post_processing,
                           dims=ndims,
                           k0=metadata["simulation0"]["k0"])

    # gen mpm inputs
    for simulation in metadata.values():
        # write mesh
        mesh_info = sim.create_mesh()
        sim.write_mesh_file(mesh_info, save_path=f"{save_path}/{simulation['name']}")

        # write particle
        particle_info = sim.create_particle(simulation["particle"])
        sim.write_particle_file(particle_info, save_path=f"{save_path}/{simulation['name']}")
        sim.plot_particle_config(particle_group_info=particle_info, save_path=f"{save_path}/{simulation['name']}")

        if k0 is not None:
            # write particle stress
            sim.particle_K0_stress(
                density=density, particle_group_info=particle_info, save_path=f"{save_path}/{simulation['name']}")

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
            analysis=analysis_resume,
            resume=True)


if __name__ == '__main__':
    app.run(main)
