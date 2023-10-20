import json
import random
import os
import utils
import argparse
from mpm_input_utils import ColumnSimulation

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="inputs.json", help="Input json file name")
args = parser.parse_args()

# Inputs
input_path = args.input_path
follow_taichi_coord = True
f = open(input_path)
inputs = json.load(f)
f.close()

ndims = inputs['ndims']
save_path = inputs['save_path']
simulation_case = inputs['simulation_case']

if inputs["gen_cube_from_data"]["generate"] == inputs["gen_cube_randomly"]["generate"]:
    raise ValueError(
        "Cube generation method should either be one of `gen_cube_from_data` or `gen_cube_randomly`")

if inputs["gen_cube_from_data"]["generate"]:
    input_metadata_path = inputs["gen_cube_from_data"]["metadata_path"]
    data_id_range = inputs['data_id_range']
    data_tag = [str(tag) for tag in range(data_id_range[0], data_id_range[1])]
    k0 = None

elif inputs["gen_cube_randomly"]["generate"]:
    # Define the range of output id list
    data_id_range = inputs['data_id_range']
    data_tag = [str(tag) for tag in range(data_id_range[0], data_id_range[1])]
    trajectory_names = [f"{simulation_case}{i}" for i in data_tag]

    # Inputs about mesh
    mesh_inputs = inputs['gen_cube_randomly']['sim_inputs']['mesh']
    simulation_domain = mesh_inputs['simulation_domain']
    ncells_per_dim = mesh_inputs['ncells_per_dim']
    # thickness of the outermost cells added on the perimeter of simulation domain
    outer_cell_thickness = mesh_inputs['outer_cell_thickness']

    # Inputs about particles
    particle_inputs = inputs['gen_cube_randomly']['sim_inputs']['particle']
    nparticle_perdim_percell = particle_inputs['nparticle_perdim_percell']
    particle_randomness = particle_inputs['particle_randomness']
    num_particle_groups = particle_inputs['num_particle_groups']
    # material id associated with each particle group
    material_id = particle_inputs['material_id']
    # particle gen candidate area
    particle_gen_candidate_area = particle_inputs['particle_gen_candidate_area']
    # length of cube for x, y dir
    particle_length = particle_inputs['particle_length']
    # magnitude of randomness for varying particle length
    range_randomness = particle_inputs['range_randomness']
    # random initial velocity range [lower_bound, upper_bound] for each dim.
    vel_bound = particle_inputs['vel_bound']

    # Other inputs
    wall_friction = inputs['wall_friction']
    # for computing geostatic stress for particles. Set `None` if not considering it
    k0 = inputs['k0']

    # Dimensionality check
    dim_check_list = [
        len(simulation_domain), len(ncells_per_dim),
        len(particle_length), len(particle_gen_candidate_area), len(vel_bound), len(vel_bound)]
    if all(dim != ndims for dim in dim_check_list):
        raise ValueError(f"Check if dimensionality of inputs comply with {ndims}")
    if len(material_id) is not num_particle_groups:
        raise ValueError(f"`num_particle_groups` should match {len(material_id)}")

else:
    raise ValueError("gen_cube type is not defined properly")

# Define materials
materials = inputs['mpm_inputs']['materials']
for material in materials:
    material["type"] = "MohrCoulomb3D" if ndims == 3 else "MohrCoulomb2D"

# Simulation options
analysis = inputs['mpm_inputs']['analysis']
analysis["type"] = "MPMExplicit3D" if ndims == 3 else "MPMExplicit2D"
analysis["resume"]["uuid"] = f"{simulation_case}"
analysis["uuid"] = f"{simulation_case}"

# Simulation options for resuming
analysis_resume = inputs['mpm_inputs']['analysis_resume']
analysis_resume["type"] = "MPMExplicit3D" if ndims == 3 else "MPMExplicit2D"
analysis_resume["resume"]["uuid"] = f"{simulation_case}"
analysis_resume["uuid"] = f"{simulation_case}"

post_processing = inputs['mpm_inputs']['post_processing']

def main():
    # Create trajectory meta data
    metadata = {}
    if inputs["gen_cube_randomly"]["generate"]:
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
            particle_ranges = utils.make_n_box_ranges(
                num_particle_groups=num_particle_groups,
                size=particle_length,
                domain=particle_gen_candidate_area,
                size_random_level=range_randomness,
                min_interval=cellsize_per_dim[0]
            )

            # assign the generated ranges, vel, and materials for each particle group
            for g in range(num_particle_groups):
                metadata[f"simulation{i}"]["particle"][f"group{g}"] = {
                    "particle_geometry": particle_ranges[g],
                    "particle_vel": [
                        random.uniform(vel_bound[i][0], vel_bound[i][1]) for i in range(ndims)],
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

    elif inputs["gen_cube_from_data"]["generate"]:
        # read metadata
        f = open(inputs["gen_cube_from_data"]["metadata_path"])
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
        input_metadata_name = inputs["gen_cube_from_data"]["metadata_path"].rsplit('/', 1)[-1]
        out_file = open(f"{save_path}/{input_metadata_name}", "w")
        json.dump(metadata, out_file, indent=2)

    else:
        raise ValueError("gen_cube type is not defined properly")

    # Init
    sim = ColumnSimulation(simulation_domain=metadata["simulation0"]["simulation_domain"],
                           ncells_per_dim=metadata["simulation0"]["ncells_per_dim"],
                           outer_cell_thickness=metadata["simulation0"]["outer_cell_thickness"],
                           npart_perdim_percell=metadata["simulation0"]["nparticle_perdim_percell"],
                           randomness=metadata["simulation0"]["particle_randomness"],
                           wall_friction=metadata["simulation0"]["wall_friction"],
                           post_processing=post_processing,
                           dims=ndims,
                           k0=metadata["simulation0"]["k0"])

    # Gen mpm inputs
    for simulation in metadata.values():
        # write mesh
        mesh_info = sim.create_mesh()
        sim.write_mesh_file(mesh_info, save_path=f"{save_path}/{simulation['name']}")

        # write particle
        particle_info = sim.create_particle(simulation["particle"])
        sim.write_particle_file(particle_info, save_path=f"{save_path}/{simulation['name']}")
        sim.plot_particle_config(particle_group_info=particle_info, save_path=f"{save_path}/{simulation['name']}")

        if k0 is not None:
            # TODO (yc): currently, K0 stress only refers to the density of the first
            #   material defined in the input json file
            density = inputs['mpm_inputs']['materials'][0]["density"]
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


if __name__ == "__main__":
    main()
