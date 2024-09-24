

This document outlines steps to set and install the challenge's python environment and other dependencies.

1. [One-Time/Installation Instructions](#onetime)
- [Prerequisites](#preq)
- [Poetry Installation](#poet)
2. [All-Time Instructions](#dev)

---
### One-Time/Installation Instructions <a name="onetime"></a>

#### Prerequisites <a name="preq"></a>
Before we begin, let's make sure we have the following prerequisites installed:

- Python (version 3.6 or higher)
- pip (Python package installer)

More information on the above can be found [here](https://www.python.org/downloads/).

Next, we will walk through the installation process of Poetry, a powerful Python package manager and dependency solver to install all libraries we need for the challenge.

#### Poetry Installation <a name="poet"></a>

To install Poetry, follow these steps:

1. Open your terminal or command prompt.

2. Run the following command to install Poetry using pip:

   ```shell
   pip install poetry
   ```
   If you encounter permission errors, you may need to run the command with administrative privileges or use a virtual environment.

3. Once the installation is complete, verify that Poetry is installed correctly by running the following command:

   ```shell
   poetry --version
   ```

   You should see the version number of Poetry printed in the terminal.

More information on managing dependencies and other Poetry commands can be found in the [official documentation](https://python-poetry.org/docs/).

4. Install our ML Challenge package and its dependencies from within this repository main directory (after downloading/cloning this repository)

   ```shell
   cd ml_challenge_as24
   poetry install
   ```

---
### All-Time Instructions <a name="dev"></a>

The above steps are needed to be done once and the outcome is a Python environment, named something like `ml-challenge-24-py3.8` . To run scripts/notebooks from a terminal, we need to activate this environment in the terminal as follows.
```shell
poetry shell
```

