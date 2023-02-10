import numpy as np
import math
import os
import json
import random
import argparse
from matplotlib import pyplot as plt
import sys
from absl import app


# TODO: save metadata to each mpm simulation folder
# TODO: initial config figure text output style improvement
# TODO: Geostatic particle stress condition

class Column2DSimulation:

    def __init__(self,
                 simulation_domain,
                 cellsize,
                 npart_perdim_percell,
                 randomness,
                 wall_friction,
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
        self.outer_cell_thickness = outer_cell_thickness
        self.npart_perdim_percell = npart_perdim_percell
        self.randomness = randomness
        self.wall_friction = wall_friction
        self.dims = 2
        self.post_processing = post_processing

    def create_mesh(self):

        # define x and y boundary of simulation
        x_bounds = self.simulation_domain[0]
        y_bounds = self.simulation_domain[1]

        # Generate mesh node coordinates
        # Generate mesh node coordinates
        xs = np.concatenate((np.array([x_bounds[0]]),
                             np.arange(x_bounds[0]+self.outer_cell_thickness, x_bounds[1]-self.outer_cell_thickness, self.cellsize),
                             np.array([x_bounds[1]-self.outer_cell_thickness, x_bounds[1]])
                             ))
        ys = np.concatenate((np.array([y_bounds[0]]),
                             np.arange(y_bounds[0]+self.outer_cell_thickness, y_bounds[1]-self.outer_cell_thickness, self.cellsize),
                             np.array([y_bounds[1]-self.outer_cell_thickness, y_bounds[1]])
                             ))

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
            entity_sets["node_sets"].append({"id": i, "set": bound_node_id[i]})

        # particle sets
        for i, (group_id, pinfo) in enumerate(particle_info.items()):
            entity_sets["particle_sets"].append(
                {"id": i, "set": np.arange(pinfo["index_range"][0], pinfo["index_range"][1] + 1).tolist()})
        print(f"Make `entity_sets.json`at {save_path}")
        with open(f"{save_path}/entity_sets.json", "w") as f:
            json.dump(entity_sets, f, indent=2)
        f.close()

    def mpm_inputfile_gen(self,
                          save_path: str,
                          material_types: list,
                          particle_info: dict,
                          analysis: dict,
                          resume=False
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
            "boundary_conditions": {}
        }
        # mpm_json["mesh"]["boundary_condition"] = {}
        # velocity constraints for boundaries
        mpm_json["mesh"]["boundary_conditions"]["velocity_constraints"] = [
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
        mpm_json["mesh"]["boundary_conditions"]["friction_constraints"] = [
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
        mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"] = []
        for i, pinfo in enumerate(particle_info.values()):
            # x_vel constraints
            mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"].append(
                {
                    "pset_id": i,
                    "dir": 0,
                    "velocity": pinfo["particle_vel"][0]
                }
            )
            # y_vel constraints
            mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"].append(
                {
                    "pset_id": i,
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
        mpm_json["analysis"] = analysis
        if resume:
            mpm_json["analysis"]["resume"]["resume"] = True

        ## Post Processing
        mpm_json["post_processing"] = self.post_processing

        print(f"Make `mpm_input.json` at {save_path}")
        if resume:
            with open(f"{save_path}/mpm_input_resume.json", "w") as f:
                json.dump(mpm_json, f, indent=2)
        else:
            with open(f"{save_path}/mpm_input.json", "w") as f:
                json.dump(mpm_json, f, indent=2)
        f.close()


import random

def make_n_box_ranges(num_particle_groups,
                      size,
                      domain,
                      size_random_level,
                      boundary_offset,
                      min_interval,
                      dimensions=2):
    """
    Generates n non-overlapping box ranges in the given domain.

    Parameters
    ----------
    num_particle_groups: int
        The number of box ranges to generate
    size: List of float
        The size of each box range in each dimension
    domain: List of tuples
        The domain in which to generate the box ranges, represented as a list of tuples
        where each tuple contains the start and end of the domain in each dimension
    size_random_level: float
        The level of randomization to apply to each box size
    boundary_offset: List of float
        The distance from the boundary to be maintained for each dimension
    min_interval: float
        The minimum interval to be maintained between boxes in each dimension
    dimensions: int, optional (default=2)
        The number of dimensions in the domain

    Returns
    -------
    boxes: List of lists
        A list of generated box ranges, represented as a list of lists, where each inner list
        contains tuples representing the start and end of the box range in each dimension.
    """
    boxes = []
    attempt = 0
    max_attempts = 100
    while len(boxes) < num_particle_groups:
        random_size = size * np.random.uniform(1 - size_random_level, 1 + size_random_level, 1)
        box = []
        for i in range(dimensions):
            start = random.uniform(
                domain[i][0]+boundary_offset[i], domain[i][1]-boundary_offset[i] - random_size[i] - min_interval)
            end = start + random_size[i]
            box.append((start, end))
        overlap = False
        for existing_box in boxes:
            overlap_count = 0
            for i in range(dimensions):
                if (existing_box[i][0] - min_interval <= box[i][1]) and (box[i][0] <= existing_box[i][1] + min_interval):
                    overlap_count += 1
                if overlap_count >= 2:
                    overlap = True
                    break
            if overlap:
                break
        if not overlap:
            boxes.append(box)
        attempt += 1
        if attempt > max_attempts:
            raise Exception(f"Could not generate non-overlapping boxes after {max_attempts} attempts")
    return boxes

