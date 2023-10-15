echo "Start Training......"

echo "RL Training"
python3 RL_controller_offline.py --RL_policy DDPG --load_model default --save_model True --Index $2 --query_type $1

echo "-----------------------------------------------------------------------------------------"

echo "Start Testing......"

echo "RL_stream_Controller"

python3 RL_stream_controller.py --RL_policy DDPG --data_file data_11 --load_model default --Index $2 --seed 0 --query_type $1
# python3 RL_stream_controller.py --RL_policy DDPG --data_file data_11 --load_model default --Index $2 --seed 1 --query_type $1
# python3 RL_stream_controller.py --RL_policy DDPG --data_file data_11 --load_model default --Index $2 --seed 2 --query_type $1

echo "End Testing......"