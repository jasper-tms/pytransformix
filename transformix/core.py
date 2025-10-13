#!/usr/bin/env python3
"""
A python wrapper for 'transformix', a command-line utility provided
as part of the package 'elastix' (https://elastix.dev/index.php)

An earlier version of this function, with an example application, can be found at
https://github.com/htem/GridTape_VNC_paper/blob/main/template_registration_pipeline/register_EM_dataset_to_template/warp_points_between_FANC_and_template.py
"""

import os
from pathlib import Path
import tempfile
import shutil
import subprocess
from typing import Union, List, Optional

import numpy as np
import npimage

default_num_threads = max(1, os.cpu_count() - 2)

Filename = Union[str, Path]


def call_transformix(command: Union[str, List[str]],
                     from_directory: Optional[Filename] = None,
                     verbose: bool = False,
                     ) -> subprocess.CompletedProcess:
    """
    Call the command-line utility 'transformix' with the given command.
    """
    if isinstance(command, str):
        command = command.split(' ')
    if not command[0] == 'transformix':
        command = ['transformix'] + command

    if not shutil.which('transformix'):
        raise FileNotFoundError(
            "'transformix' not found on the shell PATH. You must download elastix"
            " (https://github.com/SuperElastix/elastix/releases) and add its path"
            " to your shell PATH. For more detailed instructions on this, see"
            " https://github.com/jasper-tms/pytransformix/blob/main/README.md")

    # Explicitly set LIBRARY_PATH vars so that calling transformix from a python
    # script works on MacOS (info: https://github.com/htem/run_elastix/issues/3).
    # This also makes transformix work on any Linux systems where the user has
    # forgotten to set LIBRARY_PATH vars as part of installing elastix, which is
    # a common oversight since this instruction is fairly buried in the manual.
    transformix_path = os.path.realpath(shutil.which('transformix'))
    transformix_dir = os.path.dirname(transformix_path)
    # Set LD_LIBRARY_PATH and DYLD_LIBRARY_PATH to be the folder containing
    # 'transformix' as well as ../lib from that folder, which are the two
    # possible locations for the linked library file ('libANNlib'), depending
    # on if the user downloaded precompiled binaries or built from source.
    env = os.environ.copy()
    dirs_to_add = transformix_dir + ':' + os.path.dirname(transformix_dir) + '/lib'
    for var in ['LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']:
        if var in env:
            env[var] = dirs_to_add + ':' + env[var]
        else:
            env[var] = dirs_to_add

    try:
        if from_directory is not None:
            starting_dir = os.getcwd()
            os.chdir(from_directory)

        if verbose:
            print('Calling transformix with command:', command)
        # The only line in this function that actually does something.
        stdout = subprocess.run(command,
                                stdout=subprocess.PIPE,
                                env=env)
    except FileNotFoundError as e:
        if "No such file or directory: 'transformix'" in e.strerror:
            # This should never happen since we checked that transformix
            # was on the path earlier, but still checking it.
            raise FileNotFoundError(
                'transformix executable not found on shell PATH.'
                ' Is elastix installed? Is it on your PATH?')
        else:
            raise
    finally:
        if from_directory is not None:
            os.chdir(starting_dir)

    return stdout


def transform_points(points: np.ndarray,
                     transformation_file: Filename,
                     verbose: bool = False,
                     num_threads: int = default_num_threads
                     ) -> np.ndarray:
    """
    Transform points using an elastix transformation model and parameters.

    This function is a wrapper for transformix, a command-line utility provided
    as part of the package elastix (https://elastix.dev/index.php).

    Calling transformix from the command line requires points written in
    a text file in a specific odd format, and transformix's output is a
    text file in a different odd format. This function lets you simply
    provide a numpy array of points as an input and get a numpy array of
    transformed points as output, taking care of the annoying text file
    formatting that transformix requires.

    Parameters
    ----------
    points : numpy array
        An Nx3 numpy array containing N x,y,z point coordinates.
        Note that these must be in the units expected by the transformation
        file you are using. Whoever generated the transformation file should
        be able to tell you what units are expected.

    transformation_file : str or pathlib.Path
        A filename of a text file that specifies a transformation model and its
        parameters. Must be in a format recognized by the command-line utility
        'transformix'. Typically is a file generated by 'elastix'.

    Returns
    -------
    An Nx3 numpy array containing N transformed x,y,z point coordinates.
    """

    try:
        points.shape
    except AttributeError:
        points = np.array(points)

    if points.shape == (3,):
        return transform_points(points[np.newaxis, :], transformation_file)[0, :]

    def write_points_as_transformix_input_file(points, fn):
        """
        Write a numpy array containing point coordinates to file in the
        format required by transformix.
        """
        with open(fn, 'w') as f:
            f.write(f'point\n{len(points)}\n')
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

    transformation_file = str(transformation_file)
    if '/' in transformation_file:
        from_directory = os.path.dirname(transformation_file)
    else:
        from_directory = None
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare input file
        fn = os.path.join(temp_dir, 'transformix_input.txt')
        write_points_as_transformix_input_file(points, fn)
        # Run transformix
        command = ['transformix',
                   '-out', temp_dir,
                   '-tp', transformation_file,
                   '-def', fn,
                   '-threads', str(num_threads)]
        stdout = call_transformix(command, from_directory, verbose=verbose)
        # Process output file
        if verbose:
            print('Reading transformed points from file')
        output_fn = os.path.join(temp_dir, 'outputpoints.txt')
        if not os.path.exists(output_fn):
            print(stdout.stdout.decode())
            raise Exception('transformix failed, see output above for details.')
        new_pts = read_points_from_transformix_output_file(output_fn)

    return np.array(new_pts)


def transform_image(im: np.ndarray,
                    voxel_size: Union[float, List[float]],
                    transformation_file: Filename,
                    preserve_dtype: bool = True,
                    verbose: bool = False,
                    num_threads: int = default_num_threads,
                    ) -> np.ndarray:
    """
    Transform an image given as a numpy array of pixel values.

    Returns
    -------
    The transformed image as a numpy array of pixel values.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fn = os.path.join(temp_dir, 'temp_image.nrrd')
        temp_fn_transformed = os.path.join(temp_dir, 'temp_image_transformed.nrrd')
        npimage.save(im, temp_fn, pixel_size=voxel_size)
        transform_image_file(temp_fn,
                             transformation_file,
                             output_file=temp_fn_transformed,
                             verbose=verbose,
                             num_threads=num_threads)
        im_transformed = npimage.load(temp_fn_transformed)

    if preserve_dtype and im.dtype != im_transformed.dtype:
        if verbose:
            print('Casting transformed image to original dtype')
        im_transformed = npimage.cast(im_transformed, im.dtype,
                                      maximize_contrast=False)
    return im_transformed


def transform_image_file(im_file: str,
                         transformation_file: Filename,
                         output_file: Optional[Filename] = None,
                         preserve_dtype: bool = False,
                         verbose: bool = False,
                         num_threads: int = default_num_threads,
                         ) -> str:
    """
    Transform an image file.

    Returns
    -------
    The filename of the transformed image.
    """
    if preserve_dtype:
        raise NotImplementedError('preserve_dtype=True is not yet implemented.')
    transformation_file = str(transformation_file)
    if output_file is None:
        output_file = os.path.splitext(im_file)[0] + '_transformed.nrrd'
    output_file = str(output_file)
    if os.path.exists(output_file):
        raise FileExistsError(f'Output file {output_file} already exists.')

    with tempfile.TemporaryDirectory() as temp_dir:
        command = ['transformix',
                   '-out', temp_dir,
                   '-tp', transformation_file,
                   '-in', im_file,
                   '-threads', str(num_threads)]
        stdout = call_transformix(command, verbose=verbose)
        # Process output file
        output_fn = os.path.join(temp_dir, 'result.nrrd')
        if not os.path.exists(output_fn):
            print(stdout.stdout.decode())
            raise Exception('transformix failed, see output above for details.')
        # Checking again right before moving to prevent race conditions
        if os.path.exists(output_file):
            raise FileExistsError(f'Output file {output_file} already exists.')
        if verbose:
            print('Moving output file to', output_file)
        shutil.move(output_fn, output_file)

    return output_file


def create_vector_field(transformation_file: Filename, return_field: bool = True, save_field_to: Optional[Filename] = None, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Create a file corresponding to a discrete vector field from a .txt parametric transformation field using transformix.

    This function calls the 'transformix' command-line tool with the '-def all' option
    to generate a deformation field image (.nrrd) that represents the displacement
    vectors produced by the transformation.

    Parameters
    ----------
    transformation_file : str or pathlib.Path
        Path to the TransformParameters.txt file describing the transformation.
    return_field : bool
        If True, the function will return the deformation field as a NumPy array.
    save_field_to : str or pathlib.Path
        If provided, the function will save the deformation field to this path.
        The path can be relative (to the path of the transformation_file) or absolute.
        Otherwise, it will not save the field to disk.
    verbose : bool
        If True, prints additional information during processing.
    Returns
    -------
    np.ndarray
        The deformation field as a NumPy array (shape: [Z, Y, X, 3]).
    """
    if not return_field and save_field_to is None:
        raise ValueError("At least one of return_field or save_field must be True.")
    ## Resolve name of output folder
    transformation_file = Path(transformation_file)

    if save_field_to is not None:

        output_folder = Path(save_field_to)
        if not output_folder.is_absolute():
            parent_folder = os.path.dirname(os.path.abspath(transformation_file))
            output_folder = os.path.join(parent_folder, output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Build output file path: same name as transformation_file, but .nrrd
        base_name = os.path.splitext(os.path.basename(transformation_file))[0]
        output_file = os.path.join(output_folder, f"{base_name}.nrrd")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the transformix command
        command = [
            "transformix",
            "-def", "all",
            "-out", temp_dir,
            "-tp", transformation_file
        ]

        # Run transformix using the provided helper function
        stdout = call_transformix(command, verbose=verbose)

        # The expected output file from transformix
        deformation_path = os.path.join(temp_dir, "deformationField.nrrd")

        if not os.path.exists(deformation_path):
            print(stdout.stdout.decode())
            raise FileNotFoundError(
                f"Transformix did not produce deformation field: {deformation_path}"
            )

        # Either save the field
        result_path = deformation_path
        if save_field_to is not None:
            result_path = shutil.move(deformation_path, output_file)
        # or return it (or both)
        if return_field:
            return npimage.load(result_path)
        return