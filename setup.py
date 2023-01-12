import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setuptools.setup(
    name='transformix',
    version='1.0.1',
    author='Jasper Phelps',
    author_email='jasper.s.phelps@gmail.com',
    description='Python wrappers for transformix',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jasper-tms/pytransformix',
    license='GNU GPL v3',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
