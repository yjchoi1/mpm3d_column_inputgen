import random
import numpy as np

def make_n_box_ranges(num_particle_groups,
                      size,
                      domain,
                      size_random_level,
                      boundary_offset,
                      min_interval,
                      ):
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
    dimensions = len(domain)
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

