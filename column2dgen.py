import json

import numpy as np
import random
from absl import app
import os
import box_ranges_gen
from mpm_2dinput_utils import ColumnSimulation

ndims = 2
random_gen = True
if random_gen == False:
    input_metadata_name = "metadata-sand2d_train_static"
save_path = "sand2d_boundary"
simulation_case = "sand2d_train"
data_tag = [str(tag) for tag in range(0, 10)]
trajectory_names = [f"{simulation_case}{i}" for i in data_tag]
ncells_per_dim = [40, 40]  ####
outer_cell_thickness = 0.00625
simulation_domain = [[0.0, 1.0],
                     [0.0, 1.0]]
if len(simulation_domain) is not ndims:
    raise Exception("`simulation_domain` should match `ndims`")
nparticle_perdim_percell = 4
particle_randomness = 0.8
wall_friction = 0.27
k0 = 0.5  # for computing geostatic stress for particles. Set `None` if not considering it.
density = 1800  # assume all material has the same density

num_particle_groups = 1
# define materials
material_model = "MohrCoulomb3D" if ndims == 3 else "MohrCoulomb2D"
materials = [
    {
        "id": 0,
        "type": material_model,
        "density": density,
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
        "type": material_model,
        "density": density,
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
material_id = [0]  # material id associated with each particle group
if len(material_id) is not num_particle_groups:
    raise Exception("`num_particle_groups` should match len(material_id)")

if random_gen is True:
    particle_length = [0.30, 0.30]  # length of cube for x, y dir
    particle_gen_candidate_area = [[0.0, 1.0], [0.0, 0.7]]
    range_randomness = 0.1
    vel_bound = [[-2, 2], [-2, 1]]
    # error
    if len(particle_length) and len(particle_gen_candidate_area) and len(vel_bound) is not ndims:
        raise Exception("particle related inputs should match `ndims`")

# simulation options
analysis = {
    "type": "MPMExplicit3D" if ndims == 3 else "MPMExplicit2D",
    "mpm_scheme": "usf",
    "locate_particles": False,
    "dt": 1e-05,
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
    "nsteps": 105000,
    "uuid": f"{simulation_case}"

}
post_processing = {
    "path": "results/",
    "output_steps": 250,
    "vtk": ["stresses", "displacements"]
}


def main(_):
    # Create trajectory meta data
    metadata = {}
    if random_gen == True:
        for i, trajectory_name in enumerate(trajectory_names):

            # compute cellsize_per_dim and assume it is the same for all dims
            cellsize_per_dim = [
                (simulation_domain[i][1] - simulation_domain[i][0]) / ncells_per_dim[i] for i in range(ndims)]
            if not all(cellsize == cellsize_per_dim[0] for cellsize in cellsize_per_dim):
                raise NotImplementedError("All cell size per dim should be the same")

            # general simulation config
            metadata[f"simulation{i}"] = {}
            metadata[f"simulation{i}"]["name"] = trajectory_name
            metadata[f"simulation{i}"]["ncells_per_dim"] = ncells_per_dim
            metadata[f"simulation{i}"]["cellsize_perdim"] = cellsize_per_dim
            metadata[f"simulation{i}"]["outer_cell_thickness"] = outer_cell_thickness
            metadata[f"simulation{i}"]["simulation_domain"] = simulation_domain
            metadata[f"simulation{i}"]["nparticle_perdim_percell"] = nparticle_perdim_percell
            metadata[f"simulation{i}"]["particle_randomness"] = particle_randomness
            metadata[f"simulation{i}"]["k0"] = k0
            metadata[f"simulation{i}"]["wall_friction"] = wall_friction

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

        # save metadata individually for each sim folder
        for sim, siminfo in metadata.items():
            if not os.path.exists(f"{save_path}/{siminfo['name']}"):
                os.makedirs(f"{save_path}/{siminfo['name']}")
            out_file = open(f"{save_path}/{siminfo['name']}/metadata.json", "w")
            json.dump(metadata[sim], out_file, indent=2)
            out_file.close()

    # init
    sim = ColumnSimulation(simulation_domain=metadata["simulation0"]["simulation_domain"],
                           ncells_per_dim=metadata["simulation0"]["ncells_per_dim"],
                           outer_cell_thickness=metadata["simulation0"]["outer_cell_thickness"],
                           npart_perdim_percell=metadata["simulation0"]["nparticle_perdim_percell"],
                           randomness=metadata["simulation0"]["particle_randomness"],
                           wall_friction=metadata["simulation0"]["wall_friction"],
                           post_processing=post_processing,
                           dims=ndims,
                           k0=metadata["simulation0"]["particle_randomness"])

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
            analysis=analysis,
            resume=True)


if __name__ == '__main__':
    app.run(main)
