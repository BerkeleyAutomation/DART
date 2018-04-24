# DART: Noise Injection for Imitation Learning

The code is based on work by Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg.

The purpose of this repository is to make available the simulation
experiments used in the paper *DART: Noise Injection for Robust Imitation Learning*
and to provide examples of how noise injection may be used to improve off-policy imitation learning
by mitigating covariate shift.

## Requirements
Clone this repo:
	
	git clone https://github.com/BerkeleyAutomation/DART.git 
	cd DART

Create a virtual environment (optional), but useful for exact reproduction of experiments:

	virtualenv env
	source env/bin/activate

While in the VE, install the required packages:

	pip install -e .

Download mjpro131. Follow the instructions from mujoco-py for where to unzip and where to place the license key.
Again, other versions of Mujoco may work, but they have not been tested on this project.


## Reproducing the Experiments

The results from the paper with the exact same parameters can be reproduced by running the following shell scripts

	sh test.sh
	sh plot.sh

The `test.sh` script will run all four domains (Hopper, Walker, HalfCheeth, and Humanoid) for several trials using each algorithm used in the paper and in the supplementary material. Note that this may take hours to complete. Once finished, the data collected can be found in the `results/` directory. Subdirectories will be named after the parameters used.

By running `plot.sh`, reward and loss plots for each environment will be generated as in Fig. 2 and Fig. 4. Loss plots for the random covariance matrices with hand-chosen traces will also be generated as in Fig. 5 of the paper. For the loss plots, loss on the robot's distribution is shown with solid lines and error bars. Loss on the supervisor's distribution is shown with dashed lines.

The simulated error, i.e., the error simulated by a noisy supervisor, may also be plotted in a similar fashion. Although this data is collected from each test, the curves are left out by default so as not to clutter the plots.

For `plot_reward.py`, an optional `--normalize` flag may be added to normalize the reward between 0 and 1 as in the paper.

## Explanations of Experiments and Parameters

The general methods used for initializing the tasks, collecting the data and evaluating the learners can be found in `framework.py`. The number of trials to run each experiment may be specified in `framework.py`.

Each test file (`test_bc.py`, `test_dart.py`, etc.) runs a different learning algorithm which takes a series of arguments.
`test_bc.py` runs behavior cloning without any noise as a baseline. `test_dagger` runs the DAgger algorithm (Ross et al.). `test_dagger_b.py` runs DAgger-B, which is variant of DAgger where the policy is only updated on select iterations to reduce the computational burden. `test_iso.py` runs behavior cloning with a noisy supervisor with a isotropic covariance matrix. `test_rand.py` runs behavior cloning with a Gaussian-noisy supervisor where the covariance matrix is sampled from an inverse Wishart distribution and scaled to a predetermined trace. `test_dart.py` runs the DART iterative noise optimization algorithm.

Each experiment requires a series of arguments. Arguments common to all tests are given below:
	
* `--envname [string]` Name for the OpenAI gym environment e.g. Hopper-v1
* `--t [integer]` Number of times steps per trajectory
* `--iters [space-separated integers]` Iterations to evaluate the learned policy

The following are arguments specific to each algorithm:

#### DART Arguments

* `--update [space-separated integers]` Iterations to update the noise parameter.
* `--partition [integer]` Number of examples to use for noise optimizations

#### Random Noise Arguments

* `--prior [float]` Error to simulate, i.e., trace of covariance matrix of Gaussian-noisy supervisor.

#### DAgger Arguments

* `--beta [float]` Decaying probability of taking the supervisor's action during training (see Ross et al.).

#### DAgger-B Arguments

* `--update [space-separated integers]` Iterations to update the policy
* `--beta [float]` Decaying probability of taking the supervisor's action during training.

#### Isotropic Noise Arguments

* `--scale [float]` Amount to scale identity matrix

As mentioned before, the data collected from running any of these experiments will be stored in the `results/` directory under subdirectories named after the provided arguments. The data are stored as CSV files which may be extracted and plotted using `pandas` and `matplotlib` or inspected directly using any spreadsheet application. See examples of plotting code in `experiments/plot_reward.py`

