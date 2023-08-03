# CB-geo MPM Input File Generator for Granular Flow Simulation  
Generate mpm inputs for random colliding cubes in 2d or 3d box domain for [CB-geo MPM solver](https://www.cb-geo.com/research/mpm/).

# Input
### Random generation
Input files can be randomly generated. Random genernation parameters is included in the `column_gen.py`.

### Using user defined input
Input can be passed with `.json` file. Need to remove the comments for actual input.
```shell
{
   "simulation0": {
	  "name": "sand2d_frictions152",
	  "ncells_per_dim": [
	    50,
	    50
	  ],
	  "outer_cell_thickness": 0.005,  # the auxiliary cells placed outsize the outermost cells (simulation domain) for more good-looking particle behaviors at boundaries. Usually, it should be very thin (~1/4 of regular cell size)
	  "simulation_domain": [
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
	  "particle_randomness": 0.7,
	  "k0": null,
	  "wall_friction": 0.385,
	  "particle": {
	    "group0": {
	      "particle_domain": [
		[
		  0.0,
		  0.1
		],
		[
		  0.0,
		  0.1
		]
	      ],
	      "particle_vel": [
		0.5,
		0.5
	      ],
	      "material_id": 0
	    }
	  }
	},
	"simulation1": {
	  ...
    }
    ...
}
```