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

class ColumnSimulation:

    def __init__(self,
                 simulation_domain: list,
                 ncells_per_dim: list,
                 npart_perdim_percell: int,
                 randomness: float,
                 wall_friction: float,
                 post_processing: dict,
                 dims: int,
                 outer_cell_thickness: float,
                 k0: float or bool):
        """

        :param simulation_domain: Simulation domain (boundary) that you want to set.
        Note that this is not the actual simulation domain taken by MPM.
        If outer_cell_thickness is provided, the actual domain is
        [simulation_domain[0]- outer_cell_thickness, simulation_domain[1] + outer_cell_thickness]
        If not provided, the actual domain is
        [simulation_domain[0], simulation_domain[1]]
        :param ncells_per_dim: number of cells per dimension. The mesh nodes will be generated with equal space
        computed by `(simulation_domain[1] - simulation_domain[0])/ncells_per_dim`.
        :param npart_perdim_percell: number of particles to locate in each dimension of the cell
        :param randomness: uniform random distribution parameter for randomly perturb the particles
        :param wall_friction:
        :param post_processing:
        :param dims:
        :param outer_cell_thickness: If provided, the outer mesh will be added to the mesh defined by simulation_domain.
        Therefore, The actual domain is
        [simulation_domain[0]- outer_cell_thickness, simulation_domain[1] + outer_cell_thickness]
        If not provided, the actual domain is
        [simulation_domain[0], simulation_domain[1]]
        """

        self.simulation_domain = simulation_domain
        # assume cell size are the same for all dims
        cellsize_per_dim = [(simulation_domain[i][1] - simulation_domain[i][0]) / ncells_per_dim[i] for i in range(dims)]
        if not all(cellsize == cellsize_per_dim[0] for cellsize in cellsize_per_dim):
            raise NotImplementedError("All cell size per dim should be the same")
        self.cellsize = cellsize_per_dim[0]
        self.npart_perdim_percell = npart_perdim_percell
        self.randomness = randomness
        self.wall_friction = wall_friction
        self.dims = dims
        self.post_processing = post_processing
        self.k0 = k0

        mesh_coord_base = []
        if outer_cell_thickness > 0:
            for dim, coord_range in enumerate(simulation_domain):
                first_cell = np.array([coord_range[0] - outer_cell_thickness])
                end_cell = np.array([coord_range[1] + outer_cell_thickness])
                sim_base = np.linspace(coord_range[0], coord_range[1], ncells_per_dim[dim]+1)
                coord_base = np.concatenate((first_cell, sim_base, end_cell))
                mesh_coord_base.append(coord_base)
            self.mesh_coord_base = mesh_coord_base
        elif outer_cell_thickness == 0:
            for dim, coord_range in enumerate(simulation_domain):
                coord_base = np.linspace(coord_range[0], coord_range[1], ncells_per_dim[dim]+1)
                mesh_coord_base.append(coord_base)
            self.mesh_coord_base = mesh_coord_base
        else:
            raise Exception("Outer cell thickness cannot be negative value")

    def create_mesh(self):

        # For 2D
        if self.dims == 2:
            # Create node coordinates
            coords = []
            for y in self.mesh_coord_base[1]:
                for x in self.mesh_coord_base[0]:
                    coords.append([x, y])
            coords = np.array(coords)

            # Compute the number of nodes and elements for each dimension
            nnode_x = len(self.mesh_coord_base[0])
            nnode_y = len(self.mesh_coord_base[1])
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
            # Create node coordinates
            coords = np.array(
                np.meshgrid(self.mesh_coord_base[0], self.mesh_coord_base[1], self.mesh_coord_base[2])
            ).T.reshape(-1, self.dims)
            coords = coords[:, [1, 0, 2]]

            # Compute the number of nodes and elements for each dimension
            nnode_x = len(self.mesh_coord_base[0])
            nnode_y = len(self.mesh_coord_base[1])
            nnode_z = len(self.mesh_coord_base[2])
            nnode = nnode_x * nnode_y * nnode_z
            nele_x = nnode_x - 1
            nele_y = nnode_y - 1
            nele_z = nnode_z - 1
            nele = nele_x * nele_y * nele_z
            nnode_in_ele = 8

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

    def particle_K0_stress(self, density, particle_group_info, save_path):

        # TODO: make it available to assign multiple density associated with each particle set
        particle_coords = []
        for pid, pinfo in particle_group_info.items():
            coord = pinfo["particle_coords"]
            particle_coords.append(coord)
        particle_coords = np.concatenate(particle_coords)
        unit_weight = density * 9.81

        print(f"Make `particles_stresses.txt` with K0={self.k0}, density={density}")
        particle_stress = np.zeros((np.shape(particle_coords)[0], 3))  # second axis is for stress xx, yy, zz
        if self.dims == 2:
            vertical_stress = (np.max(particle_coords[: 1]) - particle_coords[:, 1]) * unit_weight  # H*Unit_Weight
            particle_stress[:, 0] = self.k0 * vertical_stress  # K0*H*Unit_Weight
            particle_stress[:, 1] = vertical_stress
            particle_stress[:, 2] = 0  # for 2d case stress zz is zero
        elif self.dims == 3:
            vertical_stress = (np.max(particle_coords[: 2]) - particle_coords[:, 2]) * unit_weight  # H*Unit_Weight
            particle_stress[:, 0] = self.k0 * vertical_stress  # K0*H*Unit_Weight
            particle_stress[:, 1] = self.k0 * vertical_stress  # K0*H*Unit_Weight
            particle_stress[:, 2] = vertical_stress
        else:
            raise NotImplementedError

        # Write the number of stressed particles
        f = open(f"{save_path}/particles-stresses.txt", "w")
        f.write(f"{np.shape(particle_coords)[0]} \n")
        f.close()

        # Write coordinates for particles
        f = open(f"{save_path}/particles-stresses.txt", "a")
        f.write(
            np.array2string(
                # particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
                particle_stress, threshold=math.inf
            ).replace(' [', '').replace('[', '').replace(']', '')
        )
        f.close()

    def plot_particle_config(self, particle_group_info, save_path):

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
            ax.set_xticks(self.mesh_coord_base[0])
            ax.set_yticks(self.mesh_coord_base[1])
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
                ax.set_xticks(self.mesh_coord_base[0])
                ax.set_yticks(self.mesh_coord_base[1])
                ax.set_zticks(self.mesh_coord_base[2])
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
                if coord[2] == z_bounds[1]:
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
            "boundary_conditions": {},
            "cell_type": "ED3H8" if self.dims == 3 else "ED2Q4",
            "isoparametric": False,
            "check_duplicates": True,
            "io_type": "Ascii3D" if self.dims == 3 else "Ascii2D",
            "node_type": "N3D" if self.dims == 3 else "N2D"
        }
        # add particle-stress if k0 is provided
        if self.k0 is not None:
            mpm_json["mesh"]["particles_stresses"] = "particles-stresses.txt"
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
            for dim in range(self.dims):
                # x_vel constraints
                mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"].append(
                    {
                        "pset_id": i,
                        "dir": dim,
                        "velocity": pinfo["particle_vel"][dim]
                    }
                )
            # # y_vel constraints
            # mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"].append(
            #     {
            #         "pset_id": i,
            #         "dir": 1,
            #         "velocity": pinfo["particle_vel"][1]
            #     }
            # )

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
        if not resume:
            mpm_json["analysis"]["resume"]["resume"] = False
        else:
            mpm_json["analysis"]["resume"]["resume"] = True
            del mpm_json["mesh"]["boundary_conditions"]["particles_velocity_constraints"]
            del mpm_json["mesh"]["particles_stresses"]

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