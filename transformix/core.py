#!/usr/bin/env python3

# A python wrapper for 'transformix', a command-line utility provided
# as part of the package 'elastix' (https://elastix.lumc.nl/index.php)

# An earlier version of this function, with an example application, can be found at 
# https://github.com/htem/GridTape_VNC_paper/blob/main/template_registration_pipeline/register_EM_dataset_to_template/warp_points_between_FANC_and_template.py

import os
import os.path
import shutil
import subprocess
import numpy as np


def transform_points(points, transformation_file):
    """
    Transform points using an elastix transformation model and parameters.

    This function is a wrapper for transformix, a command-line utility provided
    as part of the package elastix (https://elastix.lumc.nl/index.php).
    Calling transformix from the command line requires points written to a text
    file in a certain format, and transformix's output is a text file in
    another weird format. This scripts lets you simply provide a numpy array as
    an input and get a numpy array back as output, taking care of the annoying
    text file formatting that transformix requires.

    Parameters
    ---------
    points (numpy.ndarray) :
        An Nx3 numpy array representing x,y,z point coordinates.

    transformation_file (str) :
        A filename of a text file that specifies a transformation model and its
        parameters. Must be in a format recognized by the command-line utility
        'transformix'. Typically is a file generated by 'elastix'.


    Returns
    -------
    An Nx3 numpy array representing transformed x,y,z point coordinates.
    """

    def write_points_as_transformix_input_file(points, fn):
        """
        Write a numpy array to file in the format required by transformix.
        """
        with open(fn, 'w') as f:
            f.write('point\n{}\n'.format(len(points)))
            for x, y, z in points:
                f.write('%f %f %f\n'%(x, y, z))

    def read_points_from_transformix_output_file(fn):
        """
        Read points from a file generated by transformix and return
        them as a numpy array.
        """
        points = []
        with open(fn, 'r') as transformix_out:
            for line in transformix_out.readlines():
                output = line.split('OutputPoint = [ ')[1].split(' ]')[0]
                points.append([float(i) for i in output.split(' ')])
        return points

    # Make sure we can find and run the command-line utility transformix
    if not shutil.which('transformix'):
        m = ("'transformix' not found on the shell PATH. You must download elastix"
             " (https://github.com/SuperElastix/elastix/releases) and add its path"
             " to your shell PATH. For more detailed instructions on this, see"
             " https://github.com/jasper-tms/pytransformix/blob/main/README.md")
        raise FileNotFoundError(m)
    # Explicitly set LIBRARY_PATH vars so that calling transformix from a python
    # script works on MacOS. See https://github.com/htem/run_elastix/issues/3.
    # This also makes transformix work on any Linux systems where the user has
    # forgotten to set LIBRARY_PATH vars as part of installing elastix, which is
    # a common oversight since this instruction is fairly buried in the manual.
    transformix_path = os.path.realpath(shutil.which('transformix'))
    transformix_dir = os.path.dirname(transformix_path)
    # Set LD_LIBRARY_PATH and DYLD_LIBRARY_PATH to be the folder containing
    # 'transformix' as well as ../lib from that folder, which are the two
    # possible locations for the linked library file ('libANNlib'), depending
    # on if the user downloaded precompiled binaries or built from source.
    os.environ['LD_LIBRARY_PATH'] = (
        transformix_dir + ':'
        + os.path.dirname(transformix_dir) + '/lib'
    )
    os.environ['DYLD_LIBRARY_PATH'] = (
        transformix_dir + ':'
        + os.path.dirname(transformix_dir) + '/lib'
    )

    # Some types of elastix parameter files get angry if you try to use them
    # while not in their directory
    starting_dir = os.getcwd()  # Store the user's initial directory
    if '/' in transformation_file:
        os.chdir(os.path.dirname(transformation_file))  # Change directories

    # Never delete or overwrite a file without making sure the user is OK with it.
    # These are the 3 files that transformix outputs and so will overwrite if they exist.
    for fn in ['transformix_input.txt', 'outputpoints.txt', 'transformix.log']:
        if os.path.exists(fn):
            m = ('Temporary file '+fn+' already exists in '+os.getcwd()+'. '
                 'Continuing will delete it. Continue? [Y/n] ')
            if input(m).lower() != 'y':
                wd = os.getcwd()
                os.chdir(starting_dir)  # Return user to their original dir
                raise FileExistsError(wd+'/'+fn+' must be removed.')
            else:
                os.remove(fn)

    try:
        # Prepare input
        fn = 'transformix_input.txt'
        write_points_as_transformix_input_file(points, fn)

        # Create transformix command
        command = ['transformix', '-out', './',
                   '-tp', transformation_file,
                   '-def', fn]
        try:
            # Run transformix command
            m = subprocess.run(command, stdout=subprocess.PIPE)
        except FileNotFoundError as e:
            if "No such file or directory: 'transformix'" in e.strerror:
                # This should never happen since we checked that transformix
                # was on he path earlier, but still checking it.
                raise FileNotFoundError(
                    'transformix executable not found on shell PATH.'
                    ' Is elastix installed? Is it on your PATH?')
            else:
                raise

        # Process output
        if not os.path.exists('outputpoints.txt'):
            print(m.stdout.decode())
            raise Exception('transformix failed, see output above for details.')
        new_pts = read_points_from_transformix_output_file('outputpoints.txt')
    finally:
        # Clean up temporary files, and don't raise errors if they don't exist
        try: os.remove('transformix_input.txt')
        except: pass
        try: os.remove('outputpoints.txt')
        except: pass
        try: os.remove('transformix.log')
        except: pass
        # Return user to their original dir
        os.chdir(starting_dir)

    return np.array(new_pts)
