# CB-geo MPM Input File Generator for Granular Flow Simulation  
Generate mpm inputs for random colliding cubes in 2d or 3d box domain for [CB-geo MPM solver](https://www.cb-geo.com/research/mpm/).

# Input
Input files can be randomly generated. Random genernation parameters is included in the `examples/inputs.json`.

```json
{
  "ndims": 3,  
  "save_path": "examples/sand3d/", 
  "simulation_case": "sand3d",
  "data_id_range": [
    0,
    3
  ],
  "k0": null,
  "wall_friction": 0.385,
  "gen_cube_randomly": {
    "generate": false,
    "sim_inputs": {
      "mesh": {
        "simulation_domain": [
          [
            0.0,
            1.0
          ],
          [
            0.0,
            1.0
          ],
          [
            0.0,
            1.0
          ]
        ],
        "ncells_per_dim": [
          20,
          20,
          20
        ],
        "outer_cell_thickness": 0.0125
      },
      "particle": {
        "nparticle_perdim_percell": 2,
        "particle_randomness": 0.8,
        "num_particle_groups": 2,
        "material_id": [
          0,
          0
        ],
        "particle_gen_candidate_area": [
          [
            0.0,
            1.0
          ],
          [
            0.0,
            1.0
          ],
          [
            0.0,
            1.0
          ]
        ],
        "particle_length": [
          0.30,
          0.30,
          0.30
        ],
        "range_randomness": 0.1,
        "vel_bound": [
          [
            -1.5,
            1.5
          ],
          [
            -1.5,
            1.5
          ],
          [
            -1.5,
            1.5
          ]
        ]
      }
    }
  },
  "gen_cube_from_data": {
    "generate": true,
    "metadata_path": "examples/sand3d/metadata-sand3d.json"
  },
  "mpm_inputs": {
    "materials": [
      {
        "id": 0,
        "density": 1800,
        "youngs_modulus": 2000000.0,
        "poisson_ratio": 0.3,
        "friction": 42.0,
        "dilation": 0.0,
        "cohesion": 100,
        "tension_cutoff": 50,
        "softening": false,
        "peak_pdstrain": 0.0,
        "residual_friction": 42.0,
        "residual_dilation": 0.0,
        "residual_cohesion": 0.0,
        "residual_pdstrain": 0.0
      }
    ],
    "analysis": {
      "mpm_scheme": "usf",
      "locate_particles": false,
      "dt": 1e-06,
      "damping": {
        "type": "Cundall",
        "damping_factor": 0.05
      },
      "resume": {
        "resume": false,
        "step": 0
      },
      "velocity_update": false,
      "nsteps": 100000
    },
    "analysis_resume": {
      "mpm_scheme": "usf",
      "locate_particles": false,
      "dt": 1e-06,
      "damping": {
        "type": "Cundall",
        "damping_factor": 0.05
      },
      "resume": {
        "resume": true,
        "step": 0
      },
      "velocity_update": false,
      "nsteps": 100000
    },
    "post_processing": {
      "path": "results/",
      "output_steps": 2500,
      "vtk": ["displacements"]
    }
  }
}
```
### General inputs
* `ndims`: # dimensionality of the simulation.
* `save_path`: path to save outputs of the code.
* `simulation_case`: user defined name of simulation case. This is used for 
identifying the higher level output cases. 
* `data_id_range`: id range of the outputs that will be saved under. 
the name `{simulation_case}-{id}`. It follows Python range convention.
* `k0`: used for generating k0 stress.
* `wall_friction`: friction of the simulation boundaries.

### Inputs for randomly generating the mass cubes
* `generate`: options whether you do random gen or not
Mesh-related inputs
* `simulation_domain`: simulation domain, 
e.g., [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
* `ncells_per_dim`: number of cells to generate per each dimension
* `outer_cell_thickness`: thickness of the outermost cells added on the perimeter of simulation domain
Material point-related inputs
* `nparticle_perdim_percell`: number of particles per dimension in a cell. For example,
if it is 2 in three-dimensional space, $2^3=8$ material points is generated in a cell.
* `num_particle_groups`: the number of particles groups (mass cubes) to randomly generate
* `material_id`: material id associated with each particle group.
* `particle_gen_candidate_area`: particle gen candidate area, 
e.g., [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
* `particle_length`: length of cube for each dim.
* `range_randomness`: magnitude of randomness for varying particle length
* `vel_bound`:  random initial velocity range [lower_bound, upper_bound] for each dim.

### Inputs for manually generating the random mass cubes from user-defined input file
If you want to generate the mass cubes from user defined data, `gen_cube_from_data` in
the input json file can be used. Set `generate` option `true` and specify your input
file (metadata) path to `metadata_path`.

This input can be passed with `.json` file. An example is shown below.
Need to remove the comments for actual input.

```json
{
  "simulation0": {
    "name": "sand3d0",
    "ncells_per_dim": [
      20,
      20,
      20
    ],
    "outer_cell_thickness": 0.0125,
    "simulation_domain": [
      [
        0.0,
        1.0
      ],
      [
        0.0,
        1.0
      ],
      [
        0.0,
        1.0
      ]
    ],
    "nparticle_perdim_percell": 2,
    "particle_randomness": 0.8,
    "k0": null,
    "wall_friction": 0.385,
    "particle": {
      "group0": {
        "particle_geometry": [
          [
            0.02376750633275534,
            0.3139062907954934
          ],
          [
            0.24805536783250387,
            0.538194152295242
          ],
          [
            0.05619339296677366,
            0.3463321774295118
          ]
        ],
        "particle_vel": [
          0.6298567178815921,
          0.4818766015726872,
          -0.23723459156915316
        ],
        "material_id": 0
      },
      "group1": {
        "particle_geometry": [
          [
            0.5944128202556003,
            0.8760021779497826
          ],
          [
            0.49658469579313413,
            0.7781740534873165
          ],
          [
            0.40501560151497407,
            0.6866049592091565
          ]
        ],
        "particle_vel": [
          -1.3890943817767503,
          -0.8816225625110533,
          1.2106387664782687
        ],
        "material_id": 0
      }
    }
  },
  "simulation1": {
    ...
  },
  "simulation2": {
    ...
  },
  ...
}

```

### Inputs for [CB-geo MPM solver](https://www.cb-geo.com/research/mpm/).
Inputs for CB-geo MPM solver is required. This follows the same input format 
described in [CB-geo MPM docs](https://mpm.cb-geo.com/#/user/preprocess/input), so 
refer to their docs for more information. 


# Generate input
```shell
python3 gen.py --input_path="example/sand3d/inputs.json"
```

# Run MPM
* Install MPM solver based on [CB-geo MPM docs](https://mpm.cb-geo.com/#/user/compile/compile).
* Following script can be used to impose initial velocity and resume from this state
automatically. 
```shell
timeout 30s mpm -i mpm_input.json -f "${MPM_DIR}"
mpm -i mpm_input_resume.json -f "${MPM_DIR}"
```