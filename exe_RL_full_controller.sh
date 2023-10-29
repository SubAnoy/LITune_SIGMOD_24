if [ -z "$3" ]; then
    seed_value=0
else
    seed_value=$3
fi

if [ -z "$4" ]; then
    file_name=data_11
else
    file_name=$4
fi


echo "Start Training......"

echo "RL Training"

# Simple Episodic Training
python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1


#!/bin/bash
# Uncomment the following lines for Transfer RL training pipeline
# EPOCHS=100
# VALIDATION_INTERVAL=10

# for ((i=1; i<=EPOCHS; i++)); do
#     # Step 1: Initialize the DRL agent (only for the first epoch)
#     if [ "$i" -eq 1 ]; then
#         python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_0 --mode Initialization
#     fi

#     # Step 2: Update model parameters
#     python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_10 --mode training

#     # Step 3: Periodically evaluate on the validation set
#     if [ $(($i % $VALIDATION_INTERVAL)) -eq 0 ]; then

#         validation_output=$(python3 ./scripts/RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1 --data_file data_20 --mode validate | grep "VALIDATION_SCORE:")
#         validation_score=${validation_output#*:}  # This removes "VALIDATION_SCORE:" and keeps only the score

#         echo "Epoch $i: Validation Score = $validation_score"

#         # Here, you can include logic to adjust training based on validation_score
#     fi

#     # Step 4: Repeat until convergence (handled by loop)
# done



echo "-----------------------------------------------------------------------------------------"

echo "Start Testing......"

echo "RL_stream_Controller"

#Simple O2 system design
python3 ./scripts/RL_stream_controller.py --RL_policy DDPG --data_file $file_name --load_model default --Index $2 --seed $seed_value --query_type $1

#Full O2 system design
# python3 ./scripts/RL_O2.py --RL_policy DDPG --data_file $file_name --load_model default --Index $2 --seed $seed_value --query_type $1


echo "End Testing......"