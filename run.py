import torch
from torch import nn
import numpy as np
import os
import re
import time
import torch.nn.functional as F
import random
from diff_spib import DSPIB, NewTimeLaggedDataset

# Choose experiment ("three_well" or "LJ7")

experiment = "LJ7"

results_root = os.path.join("results", experiment)
os.makedirs(results_root, exist_ok=True)

existing = [d for d in os.listdir(results_root) if re.match(r"^DSPIB_V(\d+)$", d)]
nums = [int(re.search(r"V(\d+)$", d).group(1)) for d in existing]
new_version = (max(nums) + 1) if nums else 0

base_path = os.path.join(results_root, f"DSPIB_V{new_version}")
os.makedirs(base_path, exist_ok=True)

IB_path = os.path.join(base_path, "dspib")
os.makedirs(IB_path, exist_ok=True)

# Input model hyperparameters

dt = 20
z_dim = 2
encoder_type = 'Linear'
neuron_num1 = 16
neuron_num2 = 16
batch_size = 512
tolerance = 0.001
#patience = 5
#refinements = 10
#diff_patience = 50
#diff_refinements = 0
patience = 5
refinements = 5
diff_patience = 5
diff_refinements = 0
learning_rate = 0.001
beta = 1e-5
custom_var = 1
timesteps_per_sample = 16
seed = 42
tau = 0.0

hyperparams = {
    "dt": dt,
    "z_dim": z_dim,
    "encoder_type": encoder_type,
    "neuron_num1": neuron_num1,
    "neuron_num2": neuron_num2,
    "batch_size": batch_size,
    "tolerance": tolerance,
    "patience": patience,
    "refinements": refinements,
    "diff_patience": diff_patience,
    "diff_refinements": diff_refinements,
    "learning_rate": learning_rate,
    "beta": beta,
    "custom_var": custom_var,
    "timesteps_per_sample": timesteps_per_sample,
    "seed": seed,
    "tau": tau,
}

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

# Automatic data loading

if experiment == "LJ7":
    traj = np.load("datasets/LJ7_temperature_T=2_5_coordinate_number_trajs.npy", allow_pickle=True).item()
    traj = [v for v in traj.values()]
    label = np.loadtxt("datasets/LJ7_temperature_T=2_5_mu2_mu3_100kmeans_trajs.txt")
elif experiment == "three_well":
    traj = np.load("datasets/three_hole_potential_langevin_temperature_T=1_simulation_traj.npy")
    traj = [traj]
    label = np.load("datasets/three_hole_potential_langevin_temperature_T=1_initial_labels.npy")
    label = [label]
else:
    raise ValueError(f"Invalid experiment: {experiment}. Must be 'LJ7' or 'three_well'.")

data_shape = traj[0].shape[-1]
output_dim = np.max(label) + 1
output_dim = int(output_dim)

traj_mean = np.mean(np.concatenate(traj, axis=0), axis=0); traj_std = np.std(np.concatenate(traj, axis=0), axis=0)
traj_normal = []
for i in range(len(traj)):
    traj_normal.append((traj[i] - traj_mean[None, :]) / traj_std[None, :])

past_data = []; future_data = []; labels = []; dataweight = []; temp_weight = []
for i in range(len(traj_normal)):
    past_data.append(traj_normal[i][:(len(traj_normal[i])-dt)])
    future_data.append(traj_normal[i][dt:len(traj_normal[i])])
    labels.append(label[i][dt:len(traj_normal[i])])
for i in range(len(past_data)):
    dataweight.append(np.array([1 for i in range(len(past_data[i]))]))
temp_weight.append(np.array([2/2 for i in range(len(past_data[0]))]))
if experiment == "LJ7":
    temp_weight.append(np.array([5/2 for i in range(len(past_data[1]))]))

past_data = torch.from_numpy(np.concatenate(past_data, axis=0)).float().to(device) 
future_data = torch.from_numpy(np.concatenate(future_data, axis=0)).float().to(device)
labels = torch.from_numpy(np.concatenate(labels, axis=0)).to(device) 
dataweight = torch.from_numpy(np.concatenate(dataweight)).float().to(device)
temp_weight = torch.from_numpy(np.concatenate(temp_weight)).float().to(device)
label_data = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=torch.max(labels.to(torch.int64)+1)).long().to(device)

indices = list(range(len(past_data)))
split = int(np.floor(0.2 * len(past_data)))
valid_split = int(np.floor(0.8 * len(past_data)))
np.random.shuffle(indices)
train_indices, test_indices, valid_indices = indices[split:valid_split], indices[:split], indices[valid_split:]
train_dataset = NewTimeLaggedDataset(past_data[train_indices], future_data[train_indices], label_data[train_indices], dataweight[train_indices], temp_weight[train_indices])
test_dataset = NewTimeLaggedDataset(past_data[test_indices], future_data[test_indices], label_data[test_indices], dataweight[test_indices], temp_weight[test_indices])

# Create DSPIB object

IB_path = os.path.join(base_path, "dspib")

IB = DSPIB(output_dim=output_dim, data_shape=data_shape, encoder_type='Linear', z_dim=z_dim, lagtime=dt,
          beta=beta, learning_rate=learning_rate, device=device, path=IB_path, UpdateLabel=True,
          neuron_num1=neuron_num1, neuron_num2=neuron_num2, data_transform=None, custom_var=custom_var,
          timesteps_per_sample=timesteps_per_sample, tau=tau)
IB.to(device)

# By default, training involves SPIB pre-training + D-SPIB training

IB.fit(train_dataset, test_dataset, diffusion=False, batch_size=batch_size, tolerance=tolerance, patience=patience, refinements=refinements, index=seed)

IB.fit(train_dataset, test_dataset, diffusion=True, batch_size=batch_size, tolerance=tolerance, patience=diff_patience, refinements=diff_refinements, index=seed)

# For an SPIB-only training run, comment out the above command and uncomment the below command

"""
IB.fit(train_dataset, test_dataset, diffusion=False, batch_size=batch_size, tolerance=tolerance, patience=diff_patience, refinements=diff_refinements, index=seed)
"""

# Save model and hyperparameters

with open(IB_path + "_hyperparams.txt", "w") as f:
    for key, value in hyperparams.items():
        f.write(f"{key}: {value}\n")

valid_indices_path = IB_path + "_valid_indices.npy"
np.save(valid_indices_path, np.array(valid_indices))

torch.save(IB, IB_path+'_final_model.model')
