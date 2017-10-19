# Noise Injection for Imitation Learning

The code is based on work by Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg.

The purpose of this repository is to make available the simulation
experiments used in the paper *Optimizing Noise Injection for Imitation Learning*
and to provide examples of how noise injection may be used to improve off-policy imitation learning
by mitigating covariate shift.

## Requirements
Clone this repo:
	
	git clone https://github.com/BerkeleyAutomation/DART.git 
	cd noise-injection

Create a virtual environment (optional):

	virtualenv env
	source env/bin/activate

While in the VE, install the required packages:

	pip install --upgrade pip
	pip install numpy scipy matplotlib pandas sklearn
	pip install keras==2.0.4
	pip install --upgrade tfBinaryURL 

Replace tfBinaryURL with the appropriate url for your system for Tensorflow version 1.1.0 (e.g. https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py2-none-any.whl).
Note that other versions may work as well, but they have not been tested.

Clone and install gym and mujoco-py:

	git clone https://github.com/openai/gym.git
	cd gym
	pip install -e .
	pip install mujoco-py==0.5.7

Download mjpro131. Follow the instructions from mujoco-py for where to unzip and where to place the license key.
Again, other versions of Mujoco may work, but they have not been tested on this project.


## Running Tests

The general methods used for initializing the tasks, collecting the data and evaluating the learners can be found in `framework.py`.

Each test file (`test_bc.py`, `test_dart.py`, etc.) runs a different learning algorithm which takes a series of arguments.
`test_bc.py` runs behavior cloning without any noise as a baseline. `test_dagger` runs the DAgger algorithm (Ross et al.).
`test_rand.py` runs behavior cloning with a Gaussian-noisy supervisor where the covariance matrix is chosen randomly. `test_dart.py`
runs the DART iterative noise optimization algorithm.

Tests with arguments used in the DART paper are listed in `test.sh` as an example, which you may run by executing

	sh test.sh

Data from each trial will be saved as CSV files under the `results/` directory with sub-directories named after the given arguments.

Arguments common to all tests are given below:
	
* `--envname [string]` Name for the OpenAI gym environment e.g. Hopper-v1
* `--t [integer]` Number of times steps per trajectory
* `--iters [space-separated integers]` Iterations to evaluate the learned policy

#### DART Arguments

* `--update [space-separated integers]` Iterations to update the noise parameter.

#### Random Noise Arguments

* `--prior [float]` Error to simulate, i.e., trace of covariance matrix of Gaussian-noisy supervisor.

#### DAgger Arguments

* `--beta [float]` Decaying probability of taking the supervisor's action during training (see Ross et al.).

#### DAgger-B Arguments

* `--update [space-separated integers]` Iterations to update the policy
* `--beta [float]` Decaying probability of taking the supervisor's action during training.

#### Isotropic Noise Arguments

* `--scale [float]` Amount to scale identity matrix

## Plotting Results

Once the tests have finished, rewards and losses can be plotted using `plot_reward.py` and `plot_loss.py`. You may comment/uncomment sections depending on which learning algorithms you want plot. Similarly, running these scripts requires arguments in order to plot the appropriate set of tests. See `plot.sh` as an example of plotting results from tests run in `test.sh`. 

	sh plot.sh

The resulting plots can be found in `images/`

The supplementary experiments regarding random covariance matrices can be run by executing

	sh plot_rand.sh

which will generate plots comparing Behavior Cloning and DART with a covariance matrix
with known, fixed trace. The trace can be manually adjusted.


