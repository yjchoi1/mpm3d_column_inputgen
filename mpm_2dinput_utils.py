import numpy as np
import math
import os
import json
import argparse
from matplotlib import pyplot as plt
import sys
from absl import app

# inputs
# trajectory_names = []  # Path name to create mpm input files and results
# for i in range(0, 5):
#     trajectory_names.append(f"2dsand_train{i}")
# num_particle_groups = 2
# simulation_domain = [[0.0, 1.0], [0.0, 1.0]]  # simulation domain. Particle group are generated inside this domain
# cellsize = 0.025
# particle_domain = [[0.0, 1.0], [0.0, 0.7]]  # limit where the particle groups are generated.
# particle_length = [0.30, 0.30]  # dimension of particle group
# vel_bound = [-2, 2]  # lower and upper limits for random velocity vector for a particle group
# npart_perdim_percell = 4
# randomness = 0.9
# k0 = 0.5
# density = 1800


class Column2DSimulation:

    def __init__(self,
                 simulation_domain,
                 cellsize,
                 npart_perdim_percell,
                 randomness,
                 wall_friction,
                 analysis: dict,
                 post_processing: dict):
        """
        Initiate simulation
        :param simulation_domain: Simulation domain (boundary)
        :param cellsize: Size of width and height of each cell for the mesh generation
        :param npart_perdim_percell: number of particles to locate in each dimension of the cell
        :param randomness: uniform random distribution parameter for randomly perturb the particles
        """
        self.simulation_domain = simulation_domain
        self.cellsize = cellsize
        self.npart_perdim_percell = npart_perdim_percell
        self.randomness = randomness
        self.wall_friction = wall_friction
        self.dims = 2
        self.analysis = analysis
        self.post_processing = post_processing

    def create_mesh(self):

        # define x and y boundary of simulation
        x_bounds = self.simulation_domain[0]
        y_bounds = self.simulation_domain[1]

        # Generate mesh node coordinates
        xs = np.arange(x_bounds[0], x_bounds[1] + self.cellsize, self.cellsize)
        ys = np.arange(y_bounds[0], y_bounds[1] + self.cellsize, self.cellsize)
        xy = []
        for y in ys:
            for x in xs:
                xy.append([x, y])
        xy = np.array(xy)

        # Compute the number of nodes and elements for each dimension
        nnode_x = len(xs)
        nnode_y = len(ys)
        nele_x = nnode_x - 1
        nele_y = nnode_y - 1
        nnode = len(xy)
        nele = nele_x * nele_y

        # Define cell groups consisting of four nodes
        cells = np.empty((int(nele), 4))
        i = 0
        for ely in range(int(nele_y)):
            for elx in range(int(nele_x)):
                # cell index starts from 1 not 0, so there is "1+" at first
                cells[i, 0] = nnode_x * ely + elx
                cells[i, 1] = nnode_x * ely + elx + 1
                cells[i, 2] = nnode_x * (ely + 1) + elx + 1
                cells[i, 3] = nnode_x * (ely + 1) + elx
                i += 1
        cells = cells.astype(int)

        mesh_info = {"node_coords": xy,
                     "n_node_x": nnode_x,
                     "n_node_y": nnode_y,
                     "cell_groups": cells}

        return mesh_info

    def write_mesh_file(self, mesh_info, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        nnode = mesh_info["n_node_x"] * mesh_info["n_node_y"]
        nele = (mesh_info["n_node_x"] - 1) * (mesh_info["n_node_y"] - 1)

        print(f"Make `mesh.txt` at {save_path}")
        # Write the number of nodes
        f = open(f"{save_path}/mesh.txt", "w")
        f.write(f"{int(nnode)}\t{int(nele)}\n")
        f.close()

        # Append coordinate values of nodes to 'mesh.txt'
        f = open(f"{save_path}/mesh.txt", "a")
        f.write(
            np.array2string(
                mesh_info["node_coords"], separator='\t', threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
        )
        f.write('\n')
        f.close()

        # Append cell groups to 'mesh.txt'
        f = open(f"{save_path}/mesh.txt", "a")
        f.write(
            np.array2string(
                mesh_info["cell_groups"], separator='\t', threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
        )
        f.close()

#%%


    def particle_ranges(self,
                        particle_domain: list,
                        num_particle_groups: int,
                        particle_length: list,
                        boundary_offset: float,
                        range_randomness: float):
        """
        Make particle range (i.e., [[xmin, xmax], [ymin, ymax]]) within which particles are located-
        in non-overlapping places
        :param particle_domain: domain range where particle groups are generated (e.g., [[0.0, 1.0], [0.0, 0.7]])
        :param num_particle_groups: number of particle group ranges to generate
        :param particle_length: length of each dimension for the particle groups (i.e., [x_length, y_length])
        :param boundary_offset: offset value from the simulation boundaries to restrict particle group generating area
        (in order for the particles not to be too close to the boundary)
        :param range_randomness: extent that randomizes the length of the particle group
        :return: a list including ranges of particle groups
        (e.g., [[[xmin, xmax], [ymin, ymax]], [xmin, xmax], [ymin, ymax]], ...).
        These ranges does not overlap each other.
        """

        dims = 2

        def overlap(a, b):
            return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]

        def particle_range_gen(particle_length=None):
            """
            Generate particle group range [[xmin, xmax], [ymin, ymax]] in a specified restricted area
            :return:
            Particle group range [[xmin, xmax], [ymin, ymax]]
            """

            # Restrict the domain where particles are generated with the amount of specified offset from actual domain
            if particle_length is None:
                particle_length = particle_length
            restricted_pdomains = []
            for i, bound in enumerate(particle_domain):
                restricted_pmin = bound[0] + boundary_offset
                restricted_pmax = bound[1] - boundary_offset
                restricted_pdomains.append([restricted_pmin, restricted_pmax])

            # Generate a list of particle ranges (i.e., [[xmin, xman], [ymin, ymax]]) to generate particle groups -
            # in a restricted domain
            particle_length = particle_length * np.random.uniform(1-range_randomness, 1+range_randomness, 1)
            pranges = []
            for i, bound in enumerate(restricted_pdomains):
                pmin = np.round(np.random.uniform(bound[0], bound[1] - particle_length[i]), 2)
                pmax = pmin + particle_length[i]
                pranges.append([pmin, pmax])

            return pranges

        # make 3d particle ranges not to overlap each other.
        # Start with initiating the first cube object for testing
        ok_objs_ranges = []  # to save particle object ranges that pass the overlapping test
        pranges = particle_range_gen(particle_length=particle_length)
        ok_objs_ranges.append(pranges)

        # generate new cube range and test if it overlaps with the existing cubes
        while len(ok_objs_ranges) < num_particle_groups:
            # new candidate
            pranges = particle_range_gen(particle_length=particle_length)
            # test if new candidate overlaps existing range
            for test in ok_objs_ranges:
                test_result = [overlap(test[i], pranges[i]) for i in range(dims)]
            if (test_result[0] and test_result[1]):
                pass
            else:
                ok_objs_ranges.append(pranges)

        return ok_objs_ranges



    def create_particle(self, particle_meta_info: dict):
        """
        Create dict including particle coords and index range that particle gruop belongs to
        :param particle_meta_info: dict containing,
        {"particle_domain": [[xmin, xmax], [ymin, ymax]],
         "initial_vel": [vel_x, vel_y]}
        :return:
        dict containing,
        {"particle_coords": np.array([[x0, y0], [x1, y1], ..., [xn, yn]]),
         "index_range": [[start particle index of pgroup0, end particle index of pgroup0], ..., []}
        """

        particle_info = {}
        particle_index_start = 0
        # Geometry
        for i, (groud_id, ginfo) in enumerate(particle_meta_info.items()):
            px_bound = ginfo["particle_domain"][0]
            py_bound = ginfo["particle_domain"][1]
            # offset from each mesh boundary to start creating particle
            offset = self.cellsize / self.npart_perdim_percell / 2
            particle_interval = self.cellsize / self.npart_perdim_percell  # default spacing between particles
            pxmin = px_bound[0] + offset
            pxmax = px_bound[1] - offset
            pymin = py_bound[0] + offset
            pymax = py_bound[1] - offset

            # Create particle arrays
            pxs = np.arange(pxmin, pxmax + offset, particle_interval)
            pys = np.arange(pymin, pymax + offset, particle_interval)
            pxy = []
            for py in pys:
                for px in pxs:
                    pxy.append([px, py])
            pxy = np.array(pxy)
            # Disturb particle arrangement using a specified randomness
            pxy = pxy + np.random.uniform(-offset * self.randomness, offset * self.randomness, size=pxy.shape)
            # Store the particle arrays and its start and end indices for later use in creating entities sets
            particle_info[groud_id] = {
                "particle_coords": pxy,
                "material_id": ginfo["material_id"],
                "particle_vel": ginfo["particle_vel"],
                "index_range": [particle_index_start, particle_index_start + len(pxy) - 1]
            }
            particle_index_start = particle_index_start + len(pxy)

        return particle_info


    def write_particle_file(self,
                            particle_group_info: dict,
                            save_path: str):

        # Get entire particle coordinates for every particle groups
        particle_coords = []
        for pid, pinfo in particle_group_info.items():
            coord = pinfo["particle_coords"]
            particle_coords.append(coord)
        a = 3
        particle_coords = np.concatenate(particle_coords)

        # Write the number of particles
        f = open(f"{save_path}/particles.txt", "w")
        f.write(f"{particle_coords.shape[0]} \n")
        f.close()

        # Write coordinates for particles
        f = open(f"{save_path}/particles.txt", "a")
        f.write(
            np.array2string(
                # particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
                particle_coords, threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
        )
        f.close()

        # write figure of initial config
        # plot
        fig, ax = plt.subplots(tight_layout=True)
        for pinfo in particle_group_info.values():
            # plot particles
            ax.scatter(pinfo["particle_coords"][:, 0], pinfo["particle_coords"][:, 1], s=0.5)
            ax.set_xlim(self.simulation_domain[0])
            ax.set_ylim(self.simulation_domain[1])
            ax.set_aspect('equal')
            # plot velocity quiver
            x_center = \
                (pinfo["particle_coords"][:, 0].max() - pinfo["particle_coords"][:, 0].min()) / 2 + pinfo["particle_coords"][:, 0].min()
            y_center = \
                (pinfo["particle_coords"][:, 1].max() - pinfo["particle_coords"][:, 1].min()) / 2 + pinfo["particle_coords"][:, 1].min()
            ax.quiver(x_center, y_center, pinfo["particle_vel"][0], pinfo["particle_vel"][1], scale=10)
            ax.text(x_center, y_center, f"vel = {str(pinfo['particle_vel'])}")
        plt.savefig(f"{save_path}/initial_config.png")


    def write_entity(self,
                     save_path: str,
                     mesh_info: dict,
                     particle_info: dict):

        entity_sets = {
            "node_sets": [],
            "particle_sets": []
        }

        # boundary node sets
        left_bound_node_id = []
        right_bound_node_id = []
        bottom_bound_node_id = []
        upper_bound_node_id = []

        # get boundaries
        x_bounds = self.simulation_domain[0]
        y_bounds = self.simulation_domain[1]

        # node boundary entity
        for i, coord in enumerate(mesh_info["node_coords"]):
            if coord[0] == x_bounds[0]:
                left_bound_node_id.append(i)
            if coord[0] == x_bounds[1]:
                right_bound_node_id.append(i)
            if coord[1] == y_bounds[0]:
                bottom_bound_node_id.append(i)
            if coord[1] == y_bounds[1]:
                upper_bound_node_id.append(i)

        bound_node_id = [left_bound_node_id, right_bound_node_id, bottom_bound_node_id, upper_bound_node_id]

        for i in range(4):
            entity_sets["node_sets"].append({"id": f"{i}", "set": bound_node_id[i]})

        # particle sets

        for i, (group_id, pinfo) in enumerate(particle_info.items()):
            entity_sets["particle_sets"].append(
                {"id": f"{i}", "set": np.arange(pinfo["index_range"][0], pinfo["index_range"][1] + 1).tolist()})
        print("Make `entity_sets.json`")
        with open(f"{save_path}/entity_sets.json", "w") as f:
            json.dump(entity_sets, f, indent=2)
        f.close()

    def mpm_inputfile_gen(self,
                          save_path: str,
                          material_types: list,
                          particle_info: dict
                          ):
        # initiate json entry
        mpm_json = {}
        # title
        mpm_json["title"] = save_path

        ## Mesh info
        mpm_json["mesh"] = {
            "mesh": "mesh.txt",
            "entity_sets": "entity_sets.json",
            "cell_type": "ED2Q4",
            "isoparametric": False,
            "check_duplicates": True,
            "io_type": "Ascii2D",
            "node_type": "N2D",
            "boundary_condition": {}
        }
        # mpm_json["mesh"]["boundary_condition"] = {}
        # velocity constraints for boundaries
        mpm_json["mesh"]["boundary_condition"]["velocity_constraints"] = [
            {
                "nset_id": 0,  # left bound
                "dir": 0,
                "velocity": 0.0
            },
            {
                "nset_id": 1,  # right bound
                "dir": 0,
                "velocity": 0.0
            },
            {
                "nset_id": 2,  # bottom bound
                "dir": 1,
                "velocity": 0.0
            },
            {
                "nset_id": 3,  # top bound
                "dir": 1,
                "velocity": 0.0
            }
        ]
        # friction constraints for basal frictions
        mpm_json["mesh"]["boundary_condition"]["friction_constraints"] = [
            {
                "nset_id": 0,  # left bound
                "dir": 0,
                "sign_n": -1,
                "friction": self.wall_friction
            },
            {
                "nset_id": 1,  # right bound
                "dir": 0,
                "sign_n": 1,
                "friction": self.wall_friction
            },
            {
                "nset_id": 2,  # bottom bound
                "dir": 1,
                "sign_n": -1,
                "friction": self.wall_friction
            },
            {
                "nset_id": 3,  # top bound
                "dir": 1,
                "sign_n": 1,
                "friction": self.wall_friction
            }
        ]
        # particle initial velocity constraints
        mpm_json["mesh"]["boundary_condition"]["particles_velocity_constraints"] = []
        for i, pinfo in enumerate(particle_info.values()):
            # x_vel constraints
            mpm_json["mesh"]["boundary_condition"]["particles_velocity_constraints"].append(
                {
                    "pset_id": f"{i}",
                    "dir": 0,
                    "velocity": pinfo["particle_vel"][0]
                }
            )
            # y_vel constraints
            mpm_json["mesh"]["boundary_condition"]["particles_velocity_constraints"].append(
                {
                    "pset_id": f"{i}",
                    "dir": 1,
                    "velocity": pinfo["particle_vel"][1]
                }
            )

        ## Particle info
        mpm_json["particles"] = []
        for i, pinfo in enumerate(particle_info.items()):
            mpm_json["particles"].append(
                {
                    "generator": {
                        "check_duplicates": True,
                        "location": "particles.txt",
                        "io_type": "Ascii2D",
                        "pset_id": i,
                        "particle_type": "P2D",
                        "material_id": 0,
                        "type": "file"
                    }
                }
            )

        ## Materials
        mpm_json["materials"] = material_types
        mpm_json["material_sets"] = []
        for i, pinfo in enumerate(particle_info.values()):
            mpm_json["material_sets"].append(
                {
                    "material_id": pinfo["material_id"],
                    "pset_id": i
                }
            )

        ## External Loading Condition
        mpm_json["external_loading_conditions"] = {
            "gravity": [0, -9.81]
        }

        ## Analysis
        mpm_json["analysis"] = self.analysis

        ## Post Processing
        mpm_json["post_processing"] = self.post_processing

        print("Make `mpm_input.json`")
        with open(f"{save_path}/mpm_input.json", "w") as f:
            json.dump(mpm_json, f, indent=2)
        f.close()




def main(_):

    save_path = "mpm_inputs"
    trajectory_names = ["sand3d-0", "sand3d-1"]
    simulation_domain = [[0.0, 1.0], [0.0, 1.0]]
    particle_domain = [[0.0, 1.0], [0.0, 0.7]]
    cellsize = 0.025
    nparticle_perdim_percell = 4
    particle_length = [0.3, 0.3]
    randomness = 0.8
    num_particle_groups = len(trajectory_names)
    material_id = [0, 0]  #  material id of each particle group
    vel_bound = [-2.0, 2.0]
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

    # init
    sim = Column2DSimulation(simulation_domain=simulation_domain,
                             cellsize=cellsize,
                             npart_perdim_percell=nparticle_perdim_percell,
                             randomness=randomness,
                             wall_friction=0.27,
                             analysis=analysis,
                             post_processing=post_processing)

    # gen input
    for trajectory_name in trajectory_names:
        # mesh
        mesh_info = sim.create_mesh()
        sim.write_mesh_file(mesh_info, save_path=f"{save_path}/{trajectory_name}")

        # particle
        particle_meta_info = {}
        particle_ranges = sim.particle_ranges(
            particle_domain=particle_domain,
            num_particle_groups=num_particle_groups,
            particle_length=particle_length,
            boundary_offset=cellsize,
            range_randomness=0.2
        )
        for i in range(num_particle_groups):
            particle_meta_info[f"group{i}"] = {
                "particle_domain": particle_ranges[i],
                "material_id": material_id[i],
                "particle_vel":  [vel for vel in np.random.uniform(vel_bound[0], vel_bound[1], 2)]
            }
        particle_info = sim.create_particle(particle_meta_info)
        # write particle
        sim.write_particle_file(particle_info, save_path=f"{save_path}/{trajectory_name}")
        sim.write_entity(save_path=f"{save_path}/{trajectory_name}",
                     mesh_info=mesh_info,
                     particle_info=particle_info)



        # write mpm.json
        sim.mpm_inputfile_gen(
            save_path=f"{save_path}/{trajectory_name}",
            material_types=[material0],
            particle_info=particle_info)
    a = 5


    # particle
    autogen = False
    if autogen:
        pass
        # pinfo = create_particle_group()
        # create_particle_group()
        # write_particle()
    else:
        particle_info = {
            "group1":
                {"particle_domain": [[0.1, 0.3], [0.1, 0.3]],
                 "initial_velocity": [0.3, 0.7]},
            "group2":
                {"particle_domain": [[0.1, 0.4], [0.1, 0.4]],
                 "initial_velocity": [0.2, 0.5]}
            # ...
        }

if __name__ == '__main__':
    app.run(main)
