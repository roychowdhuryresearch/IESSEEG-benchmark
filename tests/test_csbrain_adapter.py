"""Montage topology supplied to the CSBrain adapter."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "csbrain"))

from csbrain_model import CHANNELS, REGIONS, TOPOLOGY, topology_order


def test_topology_is_a_permutation_of_all_channels():
    order = topology_order()
    assert len(CHANNELS) == len(REGIONS) == 19
    assert sorted(order) == list(range(19))


def test_each_channel_belongs_to_its_declared_region():
    for channel, region in zip(CHANNELS, REGIONS):
        assert channel in TOPOLOGY[region]
