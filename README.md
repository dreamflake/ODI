# Code for the object-based diverse input (ODI) method

PyTorch code for our paper

"Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input," **CVPR 2022**

*Junyoung Byun, Seungju Cho, Myung-Joon Kwon, Hee-Seon Kim, and Changick Kim*

<center>

![ODI_fig1](fig1.png)

![ODI_fig2](fig2.png)


</center>

# Getting started

## Dependencies
We have saved the conda environment for Ubuntu.

You can install the conda environment by entering the following command at the conda prompt.

> conda env create -f odi_env.yaml

## Dataset
Due to upload file size limitations, we cannot include all the test images. Instead, we included 100 test images.

You can download full test images from the following link: [Dataset for nips17 adversarial competition](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset)

Please place the images of the DEV set into './dataset/images'.

If you use full test images, please comment line 77 of 'eval_attacks.py' (-> # total_img_num=100).

## Usage

You can perform an attack experiment by entering the following command:

> python eval_attacks.py --config_idx=101

The 'eval_attacks.py' script receives config_idx as an execution argument, which designates the experiment configuration number.

The experiment configuration specifies the various parameters required for the experiment, such as epsilon, step size, various hyperparameters for ODI, etc. They are written in 'config.py,' and their summarized description is written in 'exp_info.xlsx.'

When an experiment is completed, the 'NEW_EXP_config_idx.xlsx' file is created in the results folder.
Using gen_table.py and gen_plot.py, you can print and visualize the attack success rates for comparison.