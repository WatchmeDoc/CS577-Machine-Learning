# CS577-Machine-Learning
Exercises for the CS577 Machine Learning course of the Computer Science Department of University of Crete. 


## Getting Started

The project uses Poetry for package management and Miniconda3 for virtual environment. Make sure both are installed on either your Linux/MacOS or WSL.

Create and activate a new conda environment using:

```bashrc
$ conda create -n ml_project python=3.8 && conda activate ml_project
```

Then, to install all required packages on your new environment use:

```bashrc
$ poetry install
```

Note: WSL uses a different conda than Windows. This is not a problem in general but when using an IDE, you need to somehow index PyCharm to that conda environment.

In PyCharm you can do that:
In your project in Pycharm.

* Choose File, Setting, Project, Python Interpreter, Add
* Choose WSL on the left. Linux = your Ubuntu
* Python interpreter path = `home/<your_name>/miniconda3/envs/<your_env>/bin/python3` -- this is the environment you have created in Ubuntu with Conda.