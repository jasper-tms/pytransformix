# pytransformix

A python3 wrapper for `transformix`, a command-line utility provided as part of the package [elastix](https://elastix.lumc.nl/index.php)

Currently runs on MacOS and Linux.


## Installation
### Step 1
**Either** `pip install` from [PyPI](https://pypi.org/project/transformix/):

    pip install transformix

**or** first `git clone` the repo and then `pip install` from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/pytransformix.git
    cd pytransformix
    pip install .

### Step 2
Install [elastix](https://elastix.lumc.nl/download.php) by first downloading it from the [releases page](https://github.com/SuperElastix/elastix/releases). Operating system compatibility notes:
- Ubuntu 22.04: Use elastix-5.0.1 or the latest version.
- Ubuntu 20.04: Use elastix-5.0.1 or the latest version.
- Ubuntu 18.04: Use elastix-5.0.0
- Ubuntu 16.04: Use elastix-4.9.0
- MacOS: elastix-5.0.1 worked on Big Sur, and I haven't tested other combinations. Probably safe to download the latest version.

Then extract the `.zip` or `.tar.gz` file you downloaded and put the folder somewhere on your computer. Then add that folder's `bin` subdirectory to your shell `PATH` and that folder's `lib` subdirectory to your shell `LD_LIBRARY_PATH`. For example, if you put the folder at `~/software/elastix-5.0.1-linux`, then add these three lines of text

    export PATH=$HOME/software/elastix-5.0.1-linux/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/software/elastix-5.0.1-linux/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}
    export DYLD_LIBRARY_PATH=$HOME/software/elastix-5.0.1-linux/lib${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}

to your shell config file (`~/.bashrc` for bash on Linux, `~/.bash_profile` for bash on Mac, or `~/.zshrc` for zsh on Mac). Then open up a new terminal and run `elastix`. If you see `Use "elastix --help" for information about elastix-usage.`, you're good to go. If not, feel free to [open an issue](https://github.com/jasper-tms/pytransformix/issues) and I can try to help.
