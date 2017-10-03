# Noise Injection for Imitation Learning

The purpose of this repository is to make available the simulation
experiments used in the paper *Optimizing Noise Injection for Imitation Learning*
and to provide an examples of how noise injection may be used.

## Requirements and Installation
Clone this repo:
	
	git clone git@github.com:jon--lee/noise-injection.git
	cd noise-injection

Create a virtual environment:

	virtualenv env
	source env/bin/activate

While in the VE, install the required packages:

	pip install --upgrade pip
	pip install numpy scipy matplotlib pandas sklearn
	pip install keras==2.0.4
	pip install --upgrade tfBinaryURL 

Replace tfBinaryURL with the appropriate url for your system for Tensorflow version 1.1.0 (e.g. https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py2-none-any.whl).
Note that other version may work as well, but they have not been tested.

Clone and install gym and mujoco-py:

	git clone git@github.com:openai/gym.git
	cd gym
	pip install -e .
	pip install mujoco-py==0.5.7

Download mjpro131. Follow the instructions from mujoco-py for where to unzip and where to place the license key.
Again, other versions of Mujoco may work, but they have not been tested on this project.

## Running Tests

Each test file (test_bc.py, test_dart.py, etc.) 





