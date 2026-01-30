#!/bin/bash

environment="cartpole"

machines=(
    # "cuda:0"
    # "cuda:1"
    # "cuda:2"
    "cuda:3"
    "cuda:4"
    "cuda:5"
    "cuda:6"
    "cuda:7"
)
ind_start=0
seeds=(
    1337
    42 
    0 
    24 
    7331
)
tag="without_slacks"
g_add="without_slacks"
# Function to check if a screen session containing "experiment_" is still running
is_screen_running() {
    screen -ls | grep -q "experiment_"
}
#Start a batch of experiments (multiple seeds)
for j in "${!seeds[@]}"; do
    seed=${seeds[j]}
    actual_ind=$((0 + ind_start))
    exp_name="${environment}_${seed}"
    device=${machines[j % ${#machines[@]}]}

    echo "Attempting to start screen session '${seed}_experiment_${actual_ind}' with experiment '${exp_name}' on device '${device}'"

    screen -dm -S "${seed}_experiment_${actual_ind}" bash -l -c "source ~/.bashrc && micromamba activate leapc_env && python run_sac_fop.py --env '${environment}' --seed ${seed} --device '${device}' -wg '${environment}_${g_add}' -wt '${tag}' --use-wandb --with-val"
    
    sleep 1 # I hope this is enough to keep the ordering in wandb
    if is_screen_running; then
        echo "Started experiment '${exp_name}' on environment in screen session '${seed}_experiment_${actual_ind}' with seed ${seed}, device ${device}, tag ${tag} and group ${environment}_${g_add}."
    else
        echo "Something went wrong for screen session with experiment '${exp_name}'"
    fi
done