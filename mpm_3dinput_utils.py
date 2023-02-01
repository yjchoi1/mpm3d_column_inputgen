from pycbg import preprocessing as mpminput
import numpy as np
from matplotlib import pyplot as plt
import json
import math
import os





def particle_ranges(num_particle_groups: int,
                    domain: list,
                    particle_length: list,
                    boundary_offset: float,
                    randomize_size = True,
                    dims=3):
    """

    :param num_particle_groups: num_particle_groups = 2
    :param domain: simulation domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    :param particle_length: particle_length = [0.3, 0.3, 0.3]
    :param boundary_offset: cellsize for one direction. Particle generation will be one cell away for the boundary.
    :param dims: dimension of the simulatin domain
    :return:
    """

    def overlap(a, b):
        return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]

    def particle_range_gen(domain, particle_length, randomize_size=randomize_size):
        randomness = 0.2
        if randomize_size==True:
            particle_length = particle_length * np.random.uniform(1-randomness, 1+randomness, 1)
        else:
            pass
        pranges = []
        for i, bound in enumerate(domain):
            pmin = np.round(np.random.uniform(bound[0], bound[1] - particle_length[i]), 2)
            pmax = pmin + particle_length[i]
            pranges.append([pmin, pmax])
        return pranges

    # restrict the domain where particles are generated with the amount of specified offset from actual domain
    restricted_domains = []
    for i, bound in enumerate(domain):
        restricted_pmin = bound[0] + boundary_offset
        restricted_pmax = bound[1] - boundary_offset
        restricted_domains.append([restricted_pmin, restricted_pmax])

    # make 3d particle ranges not to overlap each other.
    # Start with initiating the first cube object for testing
    ok_objs = []  # to save particle object ranges that pass the overlapping test

    # pranges = []
    # for i, bound in enumerate(restricted_domains):
    #     pmin = np.round(np.random.uniform(bound[0], bound[1] - particle_length[i]), 2)
    #     pmax = pmin + particle_length[i]
    #     pranges.append([pmin, pmax])

    pranges = particle_range_gen(domain=restricted_domains, particle_length=particle_length, randomize_size=randomize_size)
    ok_objs.append(pranges)

    # generate new cube range and test if it overlaps with the existing cubes
    while len(ok_objs) < num_particle_groups:

        # pranges = []
        # for i, bound in enumerate(domain):
        #     pmin = np.round(np.random.uniform(bound[0], bound[1] - particle_length[i]), 2)
        #     pmax = pmin + particle_length[i]
        #     pranges.append([pmin, pmax])  # new candidate

        # new candidate
        pranges = particle_range_gen(domain=restricted_domains, particle_length=particle_length, randomize_size=randomize_size)
        # test if new candidate overlaps existing range
        for test in ok_objs:
            test_result = [overlap(test[i], pranges[i]) for i in range(dims)]
        if (test_result[0] and test_result[1] and test_result[2]):
            pass
        else:
            ok_objs.append(pranges)

    return ok_objs


def mpm_input_gen(save_name, domain, cell_size, particle_info):

    ## inputs
    # save_name = "3d_sand_column"
    # cell_size = 0.1  # assume we have the same cellsize for x, y, z directions
    dim = 3
    npart_perdim_percell = 4
    particle_randomness = 0.5
    # domain = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    # init_vel=None if no initial vel, then the code will not put particle velocity constrain to the mpm input json file
    # particle_info = {
    #     "group1":
    #         {"bound": [[0.2, 0.5], [0.2, 0.5], [0.1, 0.4]], "init_vel": [1.0, 1.0, 1.0]},
    #     "group2":
    #         {"bound": [[0.6, 0.9], [0.6, 0.9], [0.2, 0.5]], "init_vel": [-1.0, -1.0, 0]}
    # }
    # material properties
    density = 1800
    phi = 30
    youngs_modulus = 2E+6
    poisson_ratio = 0.3
    friction = 30
    cohesion = 100
    residual_friction = 30
    tension_cutoff = 50
    tolerance = 1.0E-15
    boundary_friction = np.tan((1 * phi / 2) * (np.pi / 180))
    # analysis parameters
    dt = 0.00001
    nsteps = 105000
    output_step_interval = 250



    # particle array creator
    def create_particle_array(ndim: int,
                              npart_perdim_percell: int,
                              x_bound: list,
                              y_bound: list,
                              z_bound: list,
                              randomness: float,
                              cellsize: float):
        '''
        Create particle coordinate meshgrid nparray

        :param ndim: dimensions of simulation
        :param nparticle_per_dir: npart_perdim_percell
        :param x_bound: x-boundary of simulation as list (e.g., [0, 1]
        :param y_bound: similar to x-boundary
        :param z_bound: similar to x-boundary
        :param cellsize:
        :return: xyz coordinate meshgrid
        '''
        # Geometry
        offset = cellsize / npart_perdim_percell / 2
        particle_interval = cellsize / npart_perdim_percell
        xmin = x_bound[0] + offset
        xmax = x_bound[1] - offset
        ymin = y_bound[0] + offset
        ymax = y_bound[1] - offset
        zmin = z_bound[0] + offset
        zmax = z_bound[1] - offset

        # Create particle arrays
        x = np.arange(xmin, xmax + offset, particle_interval)
        y = np.arange(ymin, ymax + offset, particle_interval)
        z = np.arange(zmin, zmax + offset, particle_interval)
        xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, ndim)
        xyz = xyz + np.random.uniform(-offset * randomness, offset * randomness, size=xyz.shape)
        return xyz




    # init simulation
    ncells = []
    for i in range(dim):
        ncell_per_dir = (domain[i][1] - domain[i][0])/cell_size
        ncells.append(ncell_per_dir)
    sim = mpminput.Simulation(title=f"{save_name}")


    # simulation domain
    sim.create_mesh(
        dimensions=(domain[i][1] for i in range(dim)),
        origin=tuple([domain[i][0] for i in range(dim)]),
        ncells=tuple([ncells[i] for i in range(dim)])
    )

    # particle groups
    sim.create_particles()
    particle_groups = []
    for pinfo in particle_info.values():
        particle_group = create_particle_array(
            ndim=dim, npart_perdim_percell=npart_perdim_percell,
            x_bound=pinfo['bound'][0], y_bound=pinfo['bound'][1], z_bound=pinfo['bound'][2],
            randomness=particle_randomness, cellsize=cell_size)
        particle_groups.append(particle_group)
    particle_list = np.concatenate(particle_groups)
    sim.particles.particles = particle_list

    # define particle sets
    sim.init_entity_sets()
    particle_ginfo = {}  # just for visualization and simulation metadata
    for i, (pid, pinfo) in enumerate(particle_info.items()):
        # print(i, pinfo)
        pbound = pinfo["bound"]
        pset = sim.entity_sets.create_set(
            lambda x,y,z: pbound[0][0]<=x<=pbound[0][1] and pbound[1][0]<=y<=pbound[1][1] and pbound[2][0]<=z<=pbound[2][1],
            typ="particle")
        particle_ginfo[pid] = {
            "pset_id": pset,
            "bound": pbound,
            "particle_array": particle_groups[i],
            "initial_velocity": pinfo["init_vel"],
        }

    # The materials properties (assume all material is the same for all particle groups
    for info in particle_ginfo.values():
        # print(info['pset_id'])
        sim.materials.create_MohrCoulomb3D(pset_id=info['pset_id'],
                                           density=density,
                                           youngs_modulus=youngs_modulus,
                                           poisson_ratio=poisson_ratio,
                                           friction=friction,
                                           residual_friction=residual_friction,
                                           cohesion=cohesion,
                                           tension_cutoff=tension_cutoff,
                                           )

    # node_set and boundary condition
    walls = []
    walls.append([sim.entity_sets.create_set(lambda x, y, z: x == lim, typ="node") for lim in [0, sim.mesh.l0]])
    walls.append([sim.entity_sets.create_set(lambda x, y, z: y == lim, typ="node") for lim in [0, sim.mesh.l1]])
    walls.append([sim.entity_sets.create_set(lambda x, y, z: z == lim, typ="node") for lim in [0, sim.mesh.l2]])
    for direction, sets in enumerate(walls):
        for es in sets:
            sim.add_velocity_condition(dir=direction, vel_value=0.0, entity_set=es, typ="node")

    # wall friction
    for direction, sets in enumerate(walls):
        friction_sgn = [-1, 1]
        for i, es in enumerate(sets):
            sim.add_friction_condition(
                dir=direction, sgn_n=friction_sgn[i], frict_value=boundary_friction, node_set=es)

    # initial particle vel const
    particles_velocity_constraints = []
    for pid, pinfo in enumerate(particle_info.values()):
        if pinfo["init_vel"] is not None:
            for dir in range(dim):
                particles_velocity_constraint = {"pset_id": pid+1, "dir": dir, "velocity": pinfo["init_vel"][dir]}
                particles_velocity_constraints.append(particles_velocity_constraint)
                sim.add_velocity_condition(dir=dir, vel_value=pinfo["init_vel"][dir], entity_set=pid+1, typ="particle")

    # Other simulation parameters (gravity, number of iterations, time step, ..):
    sim.set_gravity([0, 0, -9.81])
    sim.set_analysis_parameters(dt=dt, nsteps=nsteps, mpm_scheme='usf', output_step_interval=output_step_interval)

    # write files
    sim.write_input_file()


    # Overwrite particles.txt manually (because there is an error for pycbg particle.txt output)
    particles = particle_list
    nparticles = particles.shape[0]
    print("Overwrite `particles.txt`")
    # Write the number of particles
    f = open(f"{save_name}/particles.txt", "w")
    f.write(f"{nparticles} \n")
    f.close()

    # Write coordinates for particles
    f = open(f"{save_name}/particles.txt", "a")
    f.write(
        np.array2string(
            # particles, formatter={'float_kind':lambda lam: "%.4f" % lam}, threshold=math.inf
            particles, threshold=math.inf
        ).replace(' [', '').replace('[', '').replace(']', '')
    )
    f.close()


    ## Visualization
    # visualize particle arrangement
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    simulation_info_text = {
        "simulation": {"cell_size": f"{cell_size}x{cell_size}x{cell_size}",
                       "particle_per_cell": f"{npart_perdim_percell}",
                       "domain": domain},
        "particles": []
    }
    for i, (pid, pinfo) in enumerate(particle_ginfo.items()):
        particles = pinfo["particle_array"]
        init_vel = pinfo["initial_velocity"]
        pbound = pinfo["bound"]
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], alpha=0.3, s=3)
        # show velocity quiver and value
        centers = []
        for j in range(dim):
            center = (particles[:, j].max() - particles[:, j].min()) / 2 + particles[:, j].min()
            centers.append(center)
        ax.quiver(centers[0], centers[1], centers[2], init_vel[0], init_vel[1], init_vel[2], length=0.3, color="black")

        text_loc = centers[0], centers[1], particles[:, 2].min() - domain[2][1]/10  # location of the top of a particle group
        ax.text(text_loc[0], text_loc[1], text_loc[2], f"{pid}")
        simulation_info_text["particles"].append({"pid": pid, "particle_range": pbound, "ini_vel": init_vel, "coord": str(particles)})
        ax.set_xlim(domain[0])
        ax.set_ylim(domain[1])
        ax.set_zlim(domain[2])
        ax.set_aspect('auto')
    with open(f'{save_name}/sim_metadata.json', 'w') as f:
        json.dump(simulation_info_text, f, indent=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig(f'{save_name}/initial_config.png')
    plt.show()


    # collect input files in a folder
    os.makedirs(f"./{save_name}/{save_name}", exist_ok=True)
    os.replace(f"./{save_name}/mesh.txt", f"./{save_name}/{save_name}/mesh.txt")
    os.replace(f"./{save_name}/entity_sets.json", f"./{save_name}/{save_name}/entity_sets.json")
    os.replace(f"./{save_name}/initial_config.png", f"./{save_name}/{save_name}/initial_config.png")
    os.replace(f"./{save_name}/sim_metadata.json", f"./{save_name}/{save_name}/sim_metadata.json")
    os.replace(f"./{save_name}/particles.txt", f"./{save_name}/{save_name}/particles.txt")

    # add input file vtk save
    f = open(f"./{save_name}/input_file.json")
    input_file = json.load(f)
    input_file["post_processing"]["vtk"] = ["stresses", "displacements"]
    with open(f'./{save_name}/input_file.json', 'w') as f:
        json.dump(input_file, f,  indent=4)

    # input file without initial velocity constraints to resume
    f = open(f"./{save_name}/input_file.json")
    input_file = json.load(f)
    input_file["analysis"]["resume"] = {
        "resume": True,
        "uuid": f"{save_name}",
        "step": 0
    }
    # additional edit
    del input_file["mesh"]["boundary_conditions"]["particles_velocity_constraints"]
    with open(f'./{save_name}/input_file_resume.json', 'w') as f:
        json.dump(input_file, f,  indent=4)


    #
    # ## debug
    # # read entity set
    # f = open(f"./{save_name}/{save_name}/entity_sets.json")
    # entity_sets = json.load(f)
    # print(entity_sets)
    #
    # nodeid_pycbg = []
    # for pset_dict in entity_sets['node_sets']:
    #     nodeid_pycbg.append(pset_dict["set"])
    #
    # nodes = np.array(sim.mesh.nodes)
    # x_left = np.where(nodes[:, 0] == 0)
    # x_right = np.where(nodes[:, 0] == 1)
    # y_left = np.where(nodes[:, 1] == 0)
    # y_right = np.where(nodes[:, 1] == 1)
    # z_left = np.where(nodes[:, 2] == 0)
    # z_right = np.where(nodes[:, 2] == 1)
    #
