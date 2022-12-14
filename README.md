# CS577-Machine-Learning
Exercises for the CS577 Machine Learning course of the Computer Science Department of University of Crete. 

Each folder contains the files for the respective assignment, including both the assignment description and the final report.
The datasets were also provided, and I included the files (.csv or zip with many .csv files) on them respectively. 

The exercises usually involve creating our own implementation of fundamental ML algorithms, or carrying out simulated statistical
tests (e.g. hypothesis testing). Read the respective assignment description for more information.


## Getting Started

The project uses Poetry (version >=1.2.0) for package management and Miniconda3 for virtual environment. Make sure both are installed on either your Linux/MacOS or WSL.

Create and activate a new conda environment using:

```bashrc
$ conda create -n ml_project python=3.8 -y && conda activate ml_project
```

Then, to install all required packages on your new environment use:

```bashrc
$ poetry install
```

Note: WSL uses a different conda than Windows. This is not a problem in general but when using an IDE, you need to somehow index the IDE to that conda environment.

Example, in your project in Pycharm:

* Choose File, Setting, Project, Python Interpreter, Add
* Choose WSL on the left. Linux = your Ubuntu
* Python interpreter path = `home/<your_name>/miniconda3/envs/<your_env>/bin/python3` -- this is the environment you have created in Ubuntu with Conda.

