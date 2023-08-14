#!/usr/bin/env python3

"""
Base RADICAL-EnTK application for ExaAM workflows
https://code.ornl.gov/ecpcitest/exaam/workflow
"""

import argparse
import json
import os
import sys

import radical.entk  as re
import radical.pilot as rp

BASE_PATH           = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE_DEFAULT = 'entk_config.json'

def calc_walltime(num_nodes, node_range, minutes):
    '''
       Finds the walltime that the num of nodes should fall under given a node range
       If the number of nodes is not in the range than a value of -1 is returned.
    '''
    for inode in range(1, len(node_range)):
        if (num_nodes >= node_range[inode - 1] and num_nodes < node_range[inode]):
            return int(minutes[inode - 1])
    return int(-1)

def ornl_machine_walltime_limits(resource, num_nodes):
    '''
       If ORNL resource is not here should return value of -1 else should return correct value
       as of 5/15/23.
    '''
    if "frontier" in resource:
        node_range = [0, 92, 184, 1882, 5645, 9409]
        minutes = [2 * 60, 6 * 60, 12 * 60, 12 * 60, 12 * 60]
    elif "summit" in resource:
        node_range = [0, 46, 92, 922, 2765, 4609]
        minutes = [2 * 60, 6 * 60, 12 * 60, 24 * 60, 24 * 60]
    elif "crusher" in resource:
        node_range = [0, 9, 65, 161]
        minutes = [8 * 60, 4 * 60, 2 * 60]
    elif "andes" in resource:
        node_range = [0, 17, 65, 385]
        minutes = [48 * 60, 36 * 60, 3 * 60]
    else:
        node_range = [0, 1]
        minutes = [-1]
    return calc_walltime(num_nodes, node_range, minutes)

def get_machine_walltime_limits(resource, num_nodes):
    '''
       Returns the walltime limit of a given resource when provided the number of nodes desired.
       If the resource does not have a policy defined yet then it returns a value of -1
       All walltime limits are given as integers and as minutes
    '''
    if "ornl" in resource:
        return ornl_machine_walltime_limits(resource, num_nodes)
    else:
        return int(-1)

def validate_walltime(num_nodes, cfg):
    '''
       Returns max walltime for the number of nodes requested
    '''
    resource_name = cfg['resource_description']["resource"]
    resource_walltime = cfg['resource_description']["walltime"]
    max_walltime = get_machine_walltime_limits(resource_name, num_nodes)
    if (max_walltime > -1 and resource_walltime > max_walltime):
        cfg['resource_description']["walltime"] = max_walltime
    return cfg

# ------------------------------------------------------------------------------
#
class BaseWF:

    def __init__(self, config_file=None, **kwargs):

        os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
        os.environ['RADICAL_PROFILE'] = 'TRUE'

        config_file = config_file or CONFIG_FILE_DEFAULT
        if '/' not in config_file:
            config_file = os.path.join(BASE_PATH, config_file)

        with open(config_file, encoding='utf8') as f:
            cfg = json.load(f)

        if not os.environ.get('RADICAL_PILOT_DBURL'):
            os.environ['RADICAL_PILOT_DBURL'] = cfg['mongodb']['url']

        cfg['resource_description'].update(kwargs)

        if 'num_nodes' in kwargs:
            cfg = validate_walltime(kwargs["num_nodes"], cfg)

        self._mgr = re.AppManager(reattempts=1)
        self._mgr.resource_desc = cfg['resource_description']

    def get_stages(self):
        raise NotImplementedError('Stages are not provided')

    def run(self):

        # base class represents a workflow with a single pipeline
        pipeline = re.Pipeline()
        pipeline.add_stages(self.get_stages())

        self._mgr.workflow = [pipeline]
        self._mgr.run()


# ------------------------------------------------------------------------------
#
class UQWF(BaseWF):

    def __init__(self, config_file=None, **kwargs):
        super().__init__(config_file=config_file, **kwargs)
        self.stage = re.Stage()

    def get_stages(self):
        return [self.stage]
# ------------------------------------------------------------------------------


def get_args():
    """
    Get arguments.
    :return: Arguments namespace.
    :rtype: _AttributeHolder
    """
    parser = argparse.ArgumentParser(
        description='Run the EnTK application with provided config file',
        usage='<entk app> [-c/--config <config file>]')

    parser.add_argument(
        '-c', '--config',
        dest='config_file',
        type=str,
        help='config file',
        required=False)

    return parser.parse_args(sys.argv[1:])


# ------------------------------------------------------------------------------

