# BinPackingDopamine

This repository is a quick playground for using the bin packing environment from: https://github.com/awslabs/or-rl-benchmarks

To this end we adapt some Dopamine (https://github.com/google/dopamine) agents, extending them to provide action masks, for a better learning.

Most of the code comes from the Dopamine and the OR-RL benchmark authors. This repository just glues them together.

The goal of this repo is to help our Dream Team to understand the environment by complementing the reading of the paper with easy experiments. Afterwards, adding this environment (which is just a file, as you will see) to the WorldModels can be attempted.


# Setting things up
1. Clone the project and go to that folder.

2. Create a conda environment for the project and activate it.
3. You'll need to install a couple of things 
* conda install numpy tensorflow=1.15 gym
* conda install -c powerai dopamine-rl gin-config
* conda install conda-build
* conda develop .

With this tensorflow version it might be that you need to also install some opencv packages.

4. Now you should be ready to run things.
Try running: python bin_packing_dopamine/run_evaluation.py --base_dir="test_results" --gin_files="test_configs/rainbow.gin"

Perhaps you could read the paper, and help us understand which configuration we should use for a testing which is not too time consuming.

If there is time we could also try integrating these environments with Ray (so we can also test with PPO or other models, in the not too time consuming configuration), letting us test more models. But for Ray the configuration to have a parametric model with an action mask is not so nice.

# Repo structure
/bin_packing:  Contains the environment itself, with many alternative versions of the environments within. This folder also includes baseline models proposed by the authors to solve the bin packing problem. Also important to mention, there is an __init__.py file included, which actually registers the environment path with a name into the Gym registry, so we can use it. This means that every time you import (even if you do not use it) the environment to another python file, the name will be available in the registry.
/bin_packing_dopamine: Contains our run_evaluation code snippet, which helps you to run an evaluation by stiching together configs, environments, agents and test folders.  
/bin_packing_dopamine/components: This one has code that we created on top of Dopamine. It includes parametric agents (which apply an action mask, such that the agent does not pick invalid actions), our network (defining the Keras network in use), checkpoint_runner (the instance that run_evaluation uses, to run the experiments, this is made external such that we can toy around with its aspects, if needed). 
/test_configs: Contains the gin files that define a specific experiment.
/test_results: Will hold the test results. If you want to visualize things, point tensorboard to this folder. Similarly, here the checkpointed models get saved for later use. If you run many times this might take-up some disk space.