#!/usr/bin/env python3

'''
Data helper utilities for catpro.
'''

import os

def make_output_dir(output_dir, name=''):
    path = os.path.join(output_dir, name)
    if not path.endswith('/') and not path.endswith('\\'):
        path += '/'
    os.makedirs(path, exist_ok=True)
    return path
