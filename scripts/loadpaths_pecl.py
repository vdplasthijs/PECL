'''
Use by:

import loadpaths
user_paths_dict = loadpath.loadpaths()
'''


import os
import json
import sys
import getpass
from pathlib import Path

def loadpaths(username=None):
    '''Function that loads data paths from json file based on computer username'''

    ## Get json:
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent, 'content/data_paths_pecl.json')
    json_path = str(json_path)

    if username is None:
        username = getpass.getuser()  # get username of PC account
        if username == 'runner':
            if os.uname().sysname == 'Darwin':
                username = 'runner_Darwin'
            else:
                username = 'runner_Linux'

    ## Load paths corresponding to username:
    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        assert username in config_info.keys(), f'Please add your username {username} and data paths to cnn-land-cover/data_paths.json'
        user_paths_dict = config_info[username]['paths']  # extract paths from current user
    # assert user_paths_dict['repo'] == os.getcwd(), f'Please run this script from the repo root directory. Current directory: {os.getcwd()}. Repo directory: {user_paths_dict["repo"]}'
    # Expand tildes in the json paths
    user_paths_dict = {k: str(v) for k, v in user_paths_dict.items()}
    return {k: os.path.expanduser(v) for k, v in user_paths_dict.items()}
