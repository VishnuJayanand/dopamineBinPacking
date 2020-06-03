# BinPackingDopamine

<div align="center">
  <img src="https://www.codeproject.com/KB/recipes/633133/bpp_mut_1.png"><br><br>
</div>


This repository is a quick playground for using the bin packing environment from: https://github.com/awslabs/or-rl-benchmarks

To this end we adapt some Dopamine (https://github.com/google/dopamine) agents, extending them to provide action masks, for a better learning.

<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

Most of the code comes from the Dopamine and the OR-RL benchmark authors. This repository just glues them together.

The goal of this repo is to help our Dream Team to understand the environment by complementing the reading of the paper with easy experiments. Afterwards, adding this environment (which is just a file, bin_packing/bin_packing_environment.py, and the registration in the __init__.py) to the World Models can be attempted.


# Setting things up
1. Clone the project and go to that folder.
* git clone https://github.com/gabrielcc2/BinPackingDopamine.git
* cd BinPackingDopamine

2. Create a conda environment for the project and activate it.
* conda create -n dopamine
* conda activate dopamine

3. You'll need to install a couple of things 
* conda install numpy tensorflow=1.15 gym atari_py
* conda install -c powerai dopamine-rl gin-config
* conda install conda-build
* conda develop . 

For the last instruction, notice the dot (.) at the end ;)

With this tensorflow version it might be that you need to also install some opencv packages, but I hope it will not be needed.

4. Now you should be ready to run things.
Try running: 
* python bin_packing_dopamine/run_evaluation.py --base_dir="test_results" --gin_files="test_configs/rainbow.gin"

With the configuration provided it might take about 10 minutes to run.

Try seeing the results in tensorboard, try figuring out how the baselines provided (in bin_packing/)  would fare in the same config.

As next steps perhaps you could read the paper (focusing on the bin packing experiments), and help us understand which configuration we should use for a testing which is informative, but not too time consuming. This is the configuration we should choose for the World Models.

If there is time we could also try integrating these environments with Ray (so we can also test with PPO or other models, in the not too time consuming configuration). But for Ray the configuration to have a parametric model with an action mask is not so nice ATM.

# Repo structure
* /bin_packing:  Contains the environment itself, with many alternative versions of the environments within. This folder also includes baseline models proposed by the authors to solve the bin packing problem. Also important to mention, there is an  __init__.py file included, which actually registers the environment path with a name into the Gym registry, so we can use it. This means that every time you import (even if you do not use it) the environment to another python file, the name will be available in the registry.
* /bin_packing_dopamine: Contains our run_evaluation code snippet, which helps you to run an evaluation by stiching together configs, environments, agents and test folders.  
* /bin_packing_dopamine/components: This one has code that we created on top of Dopamine. It includes parametric agents (which apply an action mask, such that the agent does not pick invalid actions), our network (defining the Keras network in use), checkpoint_runner (the instance that run_evaluation uses, to run the experiments, this is made external such that we can toy around with its aspects, if needed). 
* /test_configs: Contains the gin files that define a specific experiment.
* /test_results: Will hold the test results. If you want to visualize things, point tensorboard to this folder. Similarly, here the checkpointed models get saved for later use. If you run many times this might take-up some disk space.

<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRtvAme_kpld-DjCidCXvoX8sz0a_EpyauLKVtOIIzyGpwASYMn&usqp=CAU"><br><br>
</div>

<div align="center">
  <img src="https://salferrarello.com/wp-content/uploads/2018/06/git-push-force-not-how-it-works.jpg"><br><br>
</div>
