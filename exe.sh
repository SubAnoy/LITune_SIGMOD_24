echo "Start Testing......"

echo "Default Setting"
python3 controller.py --search_method default --data_file data_1 --Index $2 --seed 0 --query_type $1
# python3 controller.py --search_method default --data_file data_1 --Index $2 --seed 1--query_type $1
# python3 controller.py --search_method default --data_file data_1 --Index $2 --seed 2--query_type $1
wait
echo "Random Search"
python3 controller.py --search_method random_search --data_file data_1 --Index $2 --seed 0 --query_type $1
# python3 controller.py --search_method random_search --data_file data_1 --Index $2 --seed 1 --query_type $1
# python3 controller.py --search_method random_search --data_file data_1 --Index $2 --seed 2 --query_type $1
wait

echo "BO"
python3 controller.py --search_method BO --data_file data_1 --Index $2 --seed 0 --query_type $1
# python3 controller.py --search_method BO --data_file data_1 --Index $2 --seed 1 --query_type $1
# python3 controller.py --search_method BO --data_file data_1 --Index $2 --seed 2 --query_type $1
wait

echo "heuristic_search"
python3 controller.py --search_method heuristic_search --data_file data_1 --Index $2 --seed 0 --query_type $1
# python3 controller.py --search_method heuristic_search --data_file data_1 --Index $2 --seed 1 --query_type $1
# python3 controller.py --search_method heuristic_search --data_file data_1 --Index $2 --seed 2 --query_type $1
wait


echo "grid_search"
python3 controller.py --search_method grid_search --data_file data_1 --Index $2 --seed 0 --query_type $1
# python3 controller.py --search_method grid_search --data_file data_1 --Index $2 --seed 1 --query_type $1
# python3 controller.py --search_method grid_search --data_file data_1 --Index $2 --seed 2 --query_type $1
wait

echo "RL"
python3 RL_controller.py --RL_policy DDPG --data_file data_1 --save_model False --Index $2 --seed 0 --query_type $1 --load_model default
# python3 RL_controller.py --RL_policy DDPG --data_file data_1 --save_model False --Index $2 --seed 1 --query_type $1
# python3 RL_controller.py --RL_policy DDPG --data_file data_1 --save_model False --Index $2 --seed 2 --query_type $1

echo "End Testing......"


