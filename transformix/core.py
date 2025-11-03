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
import contextlib
from typing import Union, List, Optional, Iterable

import numpy as np
import npimage

default_num_threads = max(1, os.cpu_count() - 2)

Filename = Union[str, Path]

__all__ = ['call_transformix', 'change_output_settings', 'transform_points',
           'transform_image', 'transform_image_file', 'create_vector_field', 
           'get_output_settings']


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


def get_output_settings(input_file: Filename) -> dict:
    """
    Get the output settings from a TransformParameters.txt file.

    Parameters
    ----------
    input_file : str or pathlib.Path
        The filename of a TransformParameters.txt file to read.

    Returns
    -------
    dict
        A dictionary with keys 'voxel_size', 'shape', and 'origin',
        containing the corresponding output settings from the file.
    """
    settings = {}
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if line[1:8] == 'Spacing':
                settings['voxel_size'] = [float(x) for x in line.strip('()\n').split()[1:]]
            elif line[1:5] == 'Size':
                settings['shape'] = [int(x) for x in line.strip('()\n').split()[1:]]
            elif line[1:7] == 'Origin':
                settings['origin'] = [float(x) for x in line.strip('()\n').split()[1:]]
    return settings


def change_output_settings(input_file: Filename,
                           output_file: Optional[Union[Filename, 'FileWrapper']] = None,
                           voxel_size: Optional[Union[float, Iterable[float]]] = None,
                           shape: Optional[Union[int, Iterable[int]]] = None,
                           origin: Optional[Union[float, Iterable[float]]] = None,
                           overwrite: bool = False
                           ) -> Path:
    """
    Change any number of the 3 output settings in a TransformParameters.txt file.

    Parameters
    ----------
    input_file : str or pathlib.Path
        The filename of a TransformParameters.txt file to modify.

    output_file : str or pathlib.Path or object with a write() method, optional
        The file object or filename to write the modified TransformParameters to.
        If None, a new file will be created with the same filename as
          input_file but with '_modified' appended to its stem.

    voxel_size, shape, origin: float/int or list/tuple/array of 3 floats/ints, optional
        The output settings to change.
        shape must be ints, the others can be floats.
        If a single float/int is provided, it is used for all dimensions.
          e.g. voxel_size=2 -> voxel_size=(2, 2, 2) for a 3D TransformParameters file.
        If None, the setting is not changed.
        'voxel_size' corresponds the elastix parameter 'Spacing', the physical
          size (in microns or millimeters, for example) of each voxel.
        'shape' corresponds to the elastix parameter 'Size', the number of
          voxels along each dimension of the image.
        'origin' corresponds to the elastix parameter 'Origin', the physical
          location (in microns for example) where the top corner of the moving
          image is located within the fixed image's space.

    overwrite : bool, default False
        If True, overwrite output_file if it already exists.
        If False and output_file already exists, raise an error.

    Returns
    -------
    pathlib.Path
        The file path of the modified TransformParameters.
    """
    input_file = Path(input_file)
    input_file_parent = input_file.resolve().parent
    if output_file is None:
        output_file = input_file.with_stem(input_file.stem + '_modified')
        if output_file.exists() and overwrite is False:
            raise FileExistsError(f'Output file {output_file} already exists,'
                                  ' please specify a different output_file'
                                  ' or set overwrite=True.')

    # If input/output_file are already file-like objects, we skip context
    # management by using a nullcontext, and we make sure to flush manually
    if hasattr(input_file, 'readline'):
        input_file = contextlib.nullcontext(input_file)
    else:
        input_file = open(input_file, 'r')
    if hasattr(output_file, 'write'):
        output_file = contextlib.nullcontext(output_file)
        do_flush = True
    else:
        output_file = open(output_file, 'w')
        do_flush = False

    def count_entries(line: str, sep: str = ' ') -> int:
        """
        In a line like '(Spacing 2.1 1 1.4)', count how many entries
        come after the keyword, making sure to ignore multiple spaces
        between values.
        """
        while '  ' in line:
            line = line.replace(sep*2, sep)
        return len(line.strip().split(sep)) - 1

    with input_file as input_file, output_file as output_file:
        err_msg = 'Provided {} has length {}, but expected length {}.'
        for line in input_file.readlines():
            if line[1:26].lower() == 'initialtransformparameter':
                initial_transform_fn = line[line.find(' ')+1:-2].strip(' "\n')
                if initial_transform_fn.lower() != 'noinitialtransform':
                    initial_transform_fn = input_file_parent / initial_transform_fn
                    line = (f'(InitialTransformParameterFileName "{initial_transform_fn}")\n')
            elif voxel_size is not None and line[1:8] == 'Spacing':
                n_entries = count_entries(line)
                if not hasattr(voxel_size, '__len__'):
                    voxel_size = [voxel_size] * n_entries
                if len(voxel_size) != n_entries:
                    raise ValueError(err_msg.format('voxel_size', len(voxel_size), n_entries))
                line = f'(Spacing {" ".join(map(str, voxel_size))})\n'
            elif shape is not None and line[1:5] == 'Size':
                n_entries = count_entries(line)
                if not hasattr(shape, '__len__'):
                    shape = [shape] * count_entries(line)
                if len(shape) != n_entries:
                    raise ValueError(err_msg.format('shape', len(shape), n_entries))
                if not all(np.issubdtype(type(s), np.integer) for s in shape):
                    raise ValueError('All entries in "shape" must be integer types.')
                line = f'(Size {" ".join(map(str, shape))})\n'
            elif origin is not None and line[1:7] == 'Origin':
                n_entries = count_entries(line)
                if not hasattr(origin, '__len__'):
                    origin = [origin] * count_entries(line)
                if len(origin) != n_entries:
                    raise ValueError(err_msg.format('origin', len(origin), n_entries))
                line = f'(Origin {" ".join(map(str, origin))})\n'
            output_file.write(line)
        if do_flush:
            output_file.flush()

    if hasattr(output_file, 'name'):
        output_file = output_file.name
    return Path(output_file)


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
        transformix_output_fn = os.path.join(temp_dir, 'outputpoints.txt')
        if not os.path.exists(transformix_output_fn):
            print(stdout.stdout.decode())
            raise Exception('transformix failed, see output above for details.')
        new_pts = read_points_from_transformix_output_file(transformix_output_fn)

    return np.array(new_pts)


def transform_image(im: np.ndarray,
                    input_voxel_size: Union[float, List[float]],
                    transformation_file: Filename,
                    preserve_dtype: bool = True,
                    output_voxel_size: Optional[Union[float, List[float]]] = None,
                    output_shape: Optional[List[int]] = None,
                    output_origin: Optional[List[float]] = None,
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
        npimage.save(im, temp_fn, pixel_size=input_voxel_size)
        transform_image_file(temp_fn,
                             transformation_file,
                             output_file=temp_fn_transformed,
                             preserve_dtype=False,
                             output_voxel_size=output_voxel_size,
                             output_shape=output_shape,
                             output_origin=output_origin,
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
                         output_voxel_size: Optional[Union[float, List[float]]] = None,
                         output_shape: Optional[List[int]] = None,
                         output_origin: Optional[List[float]] = None,
                         overwrite: bool = False,
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
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f'Output file {output_file} already exists. Please '
                              'specify a different output_file or set overwrite=True.')

    with tempfile.TemporaryDirectory() as temp_dir, \
            tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as temp_transformation_file:
        change_output_settings(transformation_file,
                               output_file=temp_transformation_file,
                               voxel_size=output_voxel_size,
                               origin=output_origin,
                               shape=output_shape)
        command = ['transformix',
                   '-out', temp_dir,
                   '-tp', temp_transformation_file.name,
                   '-in', im_file,
                   '-threads', str(num_threads)]
        stdout = call_transformix(command, verbose=verbose)
        # Process output file
        transformix_output_fn = os.path.join(temp_dir, 'result.nrrd')
        if not os.path.exists(transformix_output_fn):
            print(stdout.stdout.decode())
            raise Exception('transformix failed, see output above for details.')
        # Check again right before moving to prevent race conditions
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(f'Output file {output_file} already exists. Please '
                                  'specify a different output_file or set overwrite=True.')
        if verbose:
            print('Moving output file to', output_file)
        shutil.move(transformix_output_fn, output_file)

    return output_file


def create_vector_field(transformation_file: Filename,
                        return_field: bool = True,
                        save_field_to: Optional[Filename] = None,
                        output_voxel_size: Optional[Union[float, List[float]]] = None,
                        output_shape: Optional[List[int]] = None,
                        output_origin: Optional[List[float]] = None,
                        overwrite: bool = False,
                        verbose: bool = False
                        ) -> Optional[np.ndarray]:
    """
    Generate a discrete vector field from a parametric transform.
    This is done by calling transformix with the `-def all` option.

    Parameters
    ----------
    transformation_file : str or pathlib.Path
        Path to the TransformParameters.txt file describing the transformation.

    return_field : bool, default True
        If True, return the deformation field as a numpy array.
        If False, return None. This option leads to slightly faster runtime.

    save_field_to : str or pathlib.Path, default None
        If None, the vector field is not saved to disk.
        If an absolute path, save the vector field there.
        If not an absolute path, it is considered relative to transformation_file.
        The path can be a directory or a filename - if it is a directory,
          the output file will have the same name as transformation_file
          but with a .nrrd extension.

    overwrite : bool, default False
        If True, overwrite an existing file at the target path.
        If False and a file already exists at the target path, raise an error.

    verbose : bool, default False
        If True, print additional information during processing.

    Returns
    -------
    np.ndarray or None
        If return_field is False, returns None.
        If return_field is True, returns the vector field as a numpy array.
        For a transformation on 3D space, the field will have shape [X, Y, Z, 3].
    """
    if not return_field and save_field_to is None:
        raise ValueError('return_field must be True or save_field must be specified.')
    transformation_file = Path(transformation_file)

    if save_field_to is not None:
        output_location = Path(save_field_to)
        if not output_location.is_absolute():
            output_location = transformation_file.parent / output_location

        if output_location.is_dir():
            os.makedirs(output_location, exist_ok=True)
            # Build output file path: same name as transformation_file, but .nrrd
            output_file = output_location / transformation_file.with_suffix('.nrrd').name
        else:
            output_file = output_location

        if output_file.exists() and not overwrite:
            raise FileExistsError(f'Output file {output_file} already exists. Please '
                                  'specify a different save_field_to or set overwrite=True.')

    with tempfile.TemporaryDirectory() as temp_dir, \
            tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as temp_transformation_file:
        change_output_settings(transformation_file,
                               output_file=temp_transformation_file,
                               voxel_size=output_voxel_size,
                               origin=output_origin,
                               shape=output_shape)
        command = ['transformix',
                   '-def', 'all',
                   '-out', temp_dir,
                   '-tp', temp_transformation_file.name]
        stdout = call_transformix(command, verbose=verbose)

        transformix_output_fn = os.path.join(temp_dir, 'deformationField.nrrd')
        if not os.path.exists(transformix_output_fn):
            print(stdout.stdout.decode())
            raise FileNotFoundError('Transformix did not produce a file at'
                                    f' {transformix_output_fn}')

        if save_field_to is not None:
            # Check again just before overwriting to prevent race condition
            # where the file was created since we last checked
            if output_file.exists() and not overwrite:
                raise FileExistsError(f'Output file {output_file} already exists. Please '
                                      'specify a different save_field_to or set overwrite=True.')
            transformix_output_fn = shutil.move(transformix_output_fn, output_file)
        if return_field:
            return npimage.load(transformix_output_fn)
