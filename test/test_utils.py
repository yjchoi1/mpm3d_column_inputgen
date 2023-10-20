import os

import pytest
import sys
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import make_n_box_ranges

def test_number_of_boxes():
    num_particle_groups = 3
    size = [2, 2]
    domain = [(0, 10), (0, 10)]
    size_random_level = 0.0
    min_interval = 1
    boxes = make_n_box_ranges(num_particle_groups, size, domain, size_random_level, min_interval)
    assert len(boxes) == num_particle_groups

def test_non_overlapping_boxes():
    num_particle_groups = 3
    size = [2, 2]
    domain = [(0, 10), (0, 10)]
    size_random_level = 0.0
    min_interval = 1
    boxes = make_n_box_ranges(num_particle_groups, size, domain, size_random_level, min_interval)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            # Check if boxes overlap. If one of these conditions is true, then they don't overlap.
            assert box1[0][1] < box2[0][0] or box1[0][0] > box2[0][1] or box1[1][1] < box2[1][0] or box1[1][0] > box2[1][1]

def test_boxes_within_domain():
    num_particle_groups = 3
    size = [2, 2]
    domain = [(0, 10), (0, 10)]
    size_random_level = 0.0
    min_interval = 1
    boxes = make_n_box_ranges(num_particle_groups, size, domain, size_random_level, min_interval)

    for box in boxes:
        for d in range(len(domain)):
            assert box[d][0] >= domain[d][0] and box[d][1] <= domain[d][1]

def test_size_random_level():
    num_particle_groups = 1
    size = [2, 2]
    domain = [(0, 10), (0, 10)]
    size_random_level = 0.5
    min_interval = 1
    box = make_n_box_ranges(num_particle_groups, size, domain, size_random_level, min_interval)[0]
    for d in range(len(size)):
        assert box[d][1] - box[d][0] >= size[d] * (1 - size_random_level)
        assert box[d][1] - box[d][0] <= size[d] * (1 + size_random_level)

def test_impossible_generation():
    num_particle_groups = 50  # unrealistic number of boxes for the small domain
    size = [2, 2]
    domain = [(0, 10), (0, 10)]
    size_random_level = 0.0
    min_interval = 1
    with pytest.raises(Exception, match=r".*Could not generate non-overlapping boxes.*"):
        make_n_box_ranges(num_particle_groups, size, domain, size_random_level, min_interval)