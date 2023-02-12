import numpy as np
import math
import os
import json
import random
import argparse
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import sys
from absl import app


# TODO: Geostatic particle stress condition

class Column2DSimulation:

    def __init__(self,
                 simulation_domain: list,
                 cellsize: float,
                 outer_cell_thickness: float,
                 npart_perdim_percell: int,
                 randomness: float,
                 wall_friction: float,
                 post_processing: dict,
                 dims: int):
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
        self.dims = dims
        self.post_processing = post_processing
        self.decimal_round = 7

    def create_mesh(self):

        # For 2D
        if self.dims == 2:
            # define x and y boundary of simulation
            coord_bases = []
            for i in range(self.dims):
                coord_base = np.concatenate(
                    (np.array([self.simulation_domain[i][0]]),
                     np.arange(round(self.simulation_domain[i][0] + self.outer_cell_thickness, self.decimal_round),
                               round(self.simulation_domain[i][1] - self.outer_cell_thickness, self.decimal_round),
                               self.cellsize),
                     np.array([round(self.simulation_domain[i][1] - self.outer_cell_thickness, self.decimal_round),
                               self.simulation_domain[i][1]])
                     )
                )
                coord_bases.append(coord_base)

            coords = []
            for y in coord_bases[1]:
                for x in coord_bases[0]:
                    coords.append([x, y])
            coords = np.array(coords)

            # Compute the number of nodes and elements for each dimension
            nnode_x = len(coord_bases[0])
            nnode_y = len(coord_bases[1])
            nele_x = nnode_x - 1
            nele_y = nnode_y - 1
            nnode = len(coords)
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

            mesh_info = {"node_coords": coords,
                         "n_node_x": nnode_x,
                         "n_node_y": nnode_y,
                         "cell_groups": cells}

        # For 3D
        else:
            # Generate mesh node coordinate bases
            coord_bases = []
            for i in range(self.dims):
                coord_base = np.concatenate(
                    (np.array([self.simulation_domain[i][0]]),
                     np.arange(round(self.simulation_domain[i][0] + self.outer_cell_thickness, self.decimal_round),
                               round(self.simulation_domain[i][1] - self.outer_cell_thickness, self.decimal_round),
                               self.cellsize),
                     np.array([round(self.simulation_domain[i][1] - self.outer_cell_thickness, self.decimal_round),
                               self.simulation_domain[i][1]])
                     )
                )
                coord_bases.append(coord_base)

            nnode_x = len(coord_bases[0])
            nnode_y = len(coord_bases[1])
            nnode_z = len(coord_bases[2])
            nnode = nnode_x * nnode_y * nnode_z
            nele_x = nnode_x - 1
            nele_y = nnode_y - 1
            nele_z = nnode_z - 1
            nele = nele_x * nele_y * nele_z
            nnode_in_ele = 8

            # Create node coordinates
            coords = np.array(np.meshgrid(coord_bases[0], coord_bases[1], coord_bases[2])).T.reshape(-1, self.dims)
            coords = coords[:, [1, 0, 2]]

            # Make cell groups
            cells = np.empty((int(nele), int(nnode_in_ele)))
            i = 0
            for elz in range(int(nele_z)):
                for ely in range(int(nele_y)):
                    for elx in range(int(nele_x)):
                        # cell index starts from 1 not 0, so there is "1+" at first
                        cells[i, 0] = nnode_x * nnode_y * elz + ely * nnode_x + elx
                        cells[i, 1] = nnode_x * nnode_y * elz + ely * nnode_x + 1 + elx
                        cells[i, 2] = nnode_x * nnode_y * elz + (ely + 1) * nnode_x + 1 + elx
                        cells[i, 3] = nnode_x * nnode_y * elz + (ely + 1) * nnode_x + elx
                        cells[i, 4] = nnode_x * nnode_y * (elz + 1) + ely * nnode_x + elx
                        cells[i, 5] = nnode_x * nnode_y * (elz + 1) + ely * nnode_x + 1 + elx
                        cells[i, 6] = nnode_x * nnode_y * (elz + 1) + (ely + 1) * nnode_x + 1 + elx
                        cells[i, 7] = nnode_x * nnode_y * (elz + 1) + (ely + 1) * nnode_x + elx
                        i += 1
            cells = cells.astype(int)

            mesh_info = {"node_coords": coords,
                         "n_node_x": nnode_x,
                         "n_node_y": nnode_y,
                         "n_node_z": nnode_z,
                         "cell_groups": cells}

        return mesh_info

    def write_mesh_file(self, mesh_info, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.dims == 3:
            nnode = mesh_info["n_node_x"] * mesh_info["n_node_y"] * mesh_info["n_node_z"]
            nele = (mesh_info["n_node_x"] - 1) * (mesh_info["n_node_y"] - 1) * (mesh_info["n_node_z"] - 1)
        else:
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
        {"particle_domain": [[xmin, xmax], [ymin, ymax], [zmin, zmax]],
         "initial_vel": [vel_x, vel_y, vel_z]}
        :return:
        dict containing,
        {"particle_coords": np.array([[x0, y0, z0], [x1, y1, z1], ..., [xn, yn, zn]]),
         "index_range": [[start particle index of pgroup0, end particle index of pgroup0], ..., []}
        """

        particle_info = {}
        particle_index_start = 0

        # Geometry
        for i, (groud_id, ginfo) in enumerate(particle_meta_info.items()):

            # particle group x, y, z range
            px_bound = ginfo["particle_domain"][0]
            py_bound = ginfo["particle_domain"][1]
            if self.dims == 3:
                pz_bound = ginfo["particle_domain"][2]

            # offset from particle gen range bound to start creating particles
            offset = self.cellsize / self.npart_perdim_percell / 2
            # spacing between particles before perturbing particle arrangement
            particle_interval = self.cellsize / self.npart_perdim_percell

            # redefine particle range
            xmin = px_bound[0] + offset
            xmax = px_bound[1] - offset
            ymin = py_bound[0] + offset
            ymax = py_bound[1] - offset
            if self.dims == 3:
                zmin = pz_bound[0] + offset
                zmax = pz_bound[1] - offset

            # Create particle range arrays
            pxs = np.arange(xmin, xmax + offset, particle_interval)
            pys = np.arange(ymin, ymax + offset, particle_interval)
            if self.dims == 3:
                pzs = np.arange(zmin, zmax + offset, particle_interval)

            # Create particle coords
            p_coords = [[px, py] for px in pxs for py in pys]
            if self.dims == 3:
                p_coords = [[px, py, pz] for px in pxs for py in pys for pz in pzs]
            p_coords = np.array(p_coords)

            # Disturb particle arrangement using a specified randomness
            p_coords = p_coords + np.random.uniform(
                -offset * self.randomness, offset * self.randomness, size=p_coords.shape)

            # Store the particle arrays and its start and end indices for later use in creating entities sets
            particle_info[groud_id] = {
                "particle_coords": p_coords,
                "material_id": ginfo["material_id"],
                "particle_vel": ginfo["particle_vel"],
                "index_range": [particle_index_start, particle_index_start + len(p_coords) - 1]
            }
            particle_index_start = particle_index_start + len(p_coords)

        return particle_info

    def write_particle_file(self,
                            particle_group_info: dict,
                            save_path: str):

        # Get entire particle coordinates for every particle groups
        particle_coords = []
        for pid, pinfo in particle_group_info.items():
            coord = pinfo["particle_coords"]
            particle_coords.append(coord)
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

    def plot_particle_config(self, particle_group_info, save_path):

        coord_bases = []
        for i in range(self.dims):
            coord_base = np.concatenate(
                (np.array([self.simulation_domain[i][0]]),
                 np.arange(self.simulation_domain[i][0] + self.outer_cell_thickness,
                           self.simulation_domain[i][1] - self.outer_cell_thickness, self.cellsize),
                 np.array([self.simulation_domain[i][1] - self.outer_cell_thickness, self.simulation_domain[i][1]])
                 ))
            coord_bases.append(coord_base)

        # write figure of initial config
        # 2d plot
        if self.dims == 2:
            fig, ax = plt.subplots(tight_layout=True)
            for i, pinfo in enumerate(particle_group_info.values()):
                # plot particles
                ax.scatter(pinfo["particle_coords"][:, 0], pinfo["particle_coords"][:, 1], s=0.5)
                ax.set_xlim(self.simulation_domain[0])
                ax.set_ylim(self.simulation_domain[1])
                ax.set_aspect('equal')
                # plot velocity quiver
                if pinfo["particle_vel"] is not None:
                    x_center = \
                        (pinfo["particle_coords"][:, 0].max() - pinfo["particle_coords"][:, 0].min()) / 2 \
                        + pinfo["particle_coords"][:, 0].min()
                    y_center = \
                        (pinfo["particle_coords"][:, 1].max() - pinfo["particle_coords"][:, 1].min()) / 2 \
                        + pinfo["particle_coords"][:, 1].min()
                    ax.quiver(x_center, y_center, pinfo["particle_vel"][0], pinfo["particle_vel"][1], scale=10)
                    print_vel = [round(vel, 2) for vel in pinfo['particle_vel']]
                    ax.text(x_center, y_center, f"vel = {str(print_vel)}")
                text_loc = [x_center, pinfo["particle_coords"][:, 1].max()]  # location of the top of a particle group
                ax.text(text_loc[0], text_loc[1], f'group{i}')
            ax.set_xticks(coord_bases[0])
            ax.set_yticks(coord_bases[1])
            # ax.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
            # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
            # ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.grid()
            plt.savefig(f"{save_path}/initial_config.png")

        # 3d plot
        elif self.dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i, pinfo in enumerate(particle_group_info.values()):
                particles = pinfo["particle_coords"]
                init_vel = pinfo["particle_vel"]
                ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], s=3.0, alpha=0.3)
                # show velocity quiver and value
                centers = []
                for j in range(self.dims):
                    center = (particles[:, j].max() - particles[:, j].min()) / 2 + particles[:, j].min()
                    centers.append(center)
                ax.quiver(centers[0], centers[1], centers[2], init_vel[0], init_vel[1], init_vel[2], length=0.3,
                          color="black")
                text_loc = centers[0], centers[1], particles[:, 2].min() - self.simulation_domain[2][
                    1] / 10  # location of the top of a particle group
                ax.text(text_loc[0], text_loc[1], text_loc[2], f"group{i}")
                ax.set_xlim(self.simulation_domain[0])
                ax.set_ylim(self.simulation_domain[1])
                ax.set_zlim(self.simulation_domain[2])
                ax.set_aspect('auto')
                ax.set_xticks(coord_bases[0])
                ax.set_yticks(coord_bases[1])
                ax.set_yticks(coord_bases[2])
                # ax.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.5)
                # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
                # ax.xaxis.set_minor_locator(MultipleLocator(5))
                ax.grid()
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
        if self.dims == 3:
            xstart_bound_node_id = []
            xend_bound_node_id = []
            ystart_bound_node_id = []
            yend_bound_node_id = []
            zstart_bound_node_id = []
            zend_bound_node_id = []
        else:
            xstart_bound_node_id = []
            xend_bound_node_id = []
            ystart_bound_node_id = []
            yend_bound_node_id = []

        # get boundaries
        if self.dims == 3:
            x_bounds = self.simulation_domain[0]
            y_bounds = self.simulation_domain[1]
            z_bounds = self.simulation_domain[2]
        else:
            x_bounds = self.simulation_domain[0]
            y_bounds = self.simulation_domain[1]

        # node boundary entity
        if self.dims == 3:
            for i, coord in enumerate(mesh_info["node_coords"]):
                if coord[0] == x_bounds[0]:
                    xstart_bound_node_id.append(i)
                if coord[0] == x_bounds[1]:
                    xend_bound_node_id.append(i)
                if coord[1] == y_bounds[0]:
                    ystart_bound_node_id.append(i)
                if coord[1] == y_bounds[1]:
                    yend_bound_node_id.append(i)
                if coord[2] == z_bounds[0]:
                    zstart_bound_node_id.append(i)
                if coord[2] == y_bounds[1]:
                    zend_bound_node_id.append(i)
        else:
            for i, coord in enumerate(mesh_info["node_coords"]):
                if coord[0] == x_bounds[0]:
                    xstart_bound_node_id.append(i)
                if coord[0] == x_bounds[1]:
                    xend_bound_node_id.append(i)
                if coord[1] == y_bounds[0]:
                    ystart_bound_node_id.append(i)
                if coord[1] == y_bounds[1]:
                    yend_bound_node_id.append(i)

        if self.dims == 3:
            bound_node_id = [xstart_bound_node_id, xend_bound_node_id,
                             ystart_bound_node_id, yend_bound_node_id,
                             zstart_bound_node_id, zend_bound_node_id]
        else:
            bound_node_id = [xstart_bound_node_id, xend_bound_node_id,
                             ystart_bound_node_id, yend_bound_node_id]

        for i in range(self.dims*2):
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
            "cell_type": "ED3H8" if self.dims == 3 else "ED2Q4",
            "isoparametric": False,
            "check_duplicates": True,
            "io_type": "Ascii3D" if self.dims == 3 else "Ascii2D",
            "node_type": "N3D" if self.dims == 3 else "N2D",
            "boundary_conditions": {}
        }
        # velocity constraints for boundaries
        if self.dims == 3:
            mpm_json["mesh"]["boundary_conditions"]["velocity_constraints"] = [
                {
                    "nset_id": 0,  # xstart bound
                    "dir": 0,
                    "velocity": 0.0
                },
                {
                    "nset_id": 1,  # xend bound
                    "dir": 0,
                    "velocity": 0.0
                },
                {
                    "nset_id": 2,  # ystart bound
                    "dir": 1,
                    "velocity": 0.0
                },
                {
                    "nset_id": 3,  # yend bound
                    "dir": 1,
                    "velocity": 0.0
                },
                {
                    "nset_id": 4,  # zstart bound
                    "dir": 2,
                    "velocity": 0.0
                },
                {
                    "nset_id": 5,  # zend bound
                    "dir": 2,
                    "velocity": 0.0
                }
            ]
        else:
            mpm_json["mesh"]["boundary_conditions"]["velocity_constraints"] = [
                {
                    "nset_id": 0,  # xstart bound
                    "dir": 0,
                    "velocity": 0.0
                },
                {
                    "nset_id": 1,  # xend bound
                    "dir": 0,
                    "velocity": 0.0
                },
                {
                    "nset_id": 2,  # ystart bound
                    "dir": 1,
                    "velocity": 0.0
                },
                {
                    "nset_id": 3,  # yend bound
                    "dir": 1,
                    "velocity": 0.0
                }
            ]
        # friction constraints for basal frictions
        if self.dims == 3:
            mpm_json["mesh"]["boundary_conditions"]["friction_constraints"] = [
                {
                    "nset_id": 0,  # xstart bound
                    "dir": 0,  # normal direction
                    "sign_n": -1,  # normal sign
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 1,  # xend bound
                    "dir": 0,
                    "sign_n": 1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 2,  # ystart bound
                    "dir": 1,
                    "sign_n": -1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 3,  # yend bound
                    "dir": 1,
                    "sign_n": 1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 4,  # zstart bound
                    "dir": 2,
                    "sign_n": -1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 5,  # zend bound
                    "dir": 2,
                    "sign_n": 1,
                    "friction": self.wall_friction
                }
            ]
        else:
            mpm_json["mesh"]["boundary_conditions"]["friction_constraints"] = [
                {
                    "nset_id": 0,  # xstart bound
                    "dir": 0,  # normal direction
                    "sign_n": -1,  # normal sign
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 1,  # xend bound
                    "dir": 0,
                    "sign_n": 1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 2,  # ystart bound
                    "dir": 1,
                    "sign_n": -1,
                    "friction": self.wall_friction
                },
                {
                    "nset_id": 3,  # yend bound
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
        for i, pinfo in enumerate(particle_info.values()):
            mpm_json["particles"].append(
                {
                    "generator": {
                        "check_duplicates": True,
                        "location": "particles.txt",
                        "io_type": "Ascii3D" if self.dims == 3 else "Ascii2D",
                        "pset_id": i,
                        "particle_type": "P3D" if self.dims == 3 else "P2D",
                        "material_id": pinfo["material_id"],
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
            "gravity": [0, 0, -9.81] if self.dims == 3 else [0, -9.81]
        }

        ## Analysis
        mpm_json["analysis"] = analysis
        if resume:
            mpm_json["analysis"]["resume"]["resume"] = True
            del mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"]

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
