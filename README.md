# pytransformix

A python3 wrapper for `transformix`, a command-line utility provided as part of the package [elastix](https://elastix.lumc.nl/index.php)

Currently runs on MacOS and Linux.


## Installation
### Step 1
**Either** `pip install` this package directly from GitHub:
 
    
    pip install git+https://github.com/jasper-tms/pytransformix.git

**or** first `git clone` it and then `pip install` it from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/pytransformix.git
    cd pytransformix
    pip install .

### Step 2
Install [elastix](https://elastix.lumc.nl/download.php) by first downloading it from the [releases page](https://github.com/SuperElastix/elastix/releases). Operating system compatibility notes:
- Ubuntu 20.04: Use elastix-5.0.1 or the latest version.
- Ubuntu 18.04: Use elastix-5.0.0
- Ubuntu 16.04: Use elastix-4.9.0
- MacOS: elastix-5.0.1 worked on Big Sur, and I haven't tested other combinations. Probably safe to download the latest version.

Then extract the `.zip` or `.tar.gz` file you downloaded and put the folder somewhere on your computer. Then add that folder's `bin` subdirectory to your shell `PATH`. For example, if you put the folder at `~/software/elastix-5.0.1-linux`, then add the line of text `export PATH=~/software/elastix-5.0.1-linux/bin` to your shell config file (`~/.bashrc` for bash on Linux,  `~/.bash_profile` for bash on Mac, or `~/.zshrc` for zsh on Mac).
