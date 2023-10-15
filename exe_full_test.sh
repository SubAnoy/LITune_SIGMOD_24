echo "Start Testing Full Data......"


# file_name=data1
file_name=$3
# file_name=osm_cellids_200M_uint64
# file_name=fb_200M_uint64
# file_name=concat_MIX_200M_uint64
# file_name=books_200M_uint64

search_budget=100

echo "Default Setting, budget $search_budget"

# python3 ./data_stream_control.py --data_file $file_name --search_method default --Index $2 --search_budget $search_budget --seed 0 --query_type $1
python3 ./data_stream_control.py --data_file $file_name --search_method default --Index $2 --search_budget $search_budget --seed 0 --query_type $1
# python3 ./data_stream_control.py --data_file $file_name --search_method default --Index $2 --search_budget $search_budget --seed 2 --query_type $1

wait

echo "Random Search, budget $search_budget"

# python3 ./data_stream_control.py --data_file $file_name --search_method random_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
python3 ./data_stream_control.py --data_file $file_name --search_method random_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
# python3 ./data_stream_control.py --data_file $file_name --search_method random_search --Index $2 --search_budget $search_budget --seed 2 --query_type $1

wait

echo "Heuristic Search, budget $search_budget"

# python3 ./data_stream_control.py --data_file $file_name --search_method heuristic_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
python3 ./data_stream_control.py --data_file $file_name --search_method heuristic_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
# python3 ./data_stream_control.py --data_file $file_name --search_method heuristic_search --Index $2 --search_budget $search_budget --seed 2 --query_type $1

wait

echo "BO, budget $search_budget"

# python3 ./data_stream_control.py --data_file $file_name --search_method BO --Index $2 --search_budget $search_budget --seed 0 --query_type $1
python3 ./data_stream_control.py --data_file $file_name --search_method BO --Index $2 --search_budget $search_budget --seed 0 --query_type $1
# python3 ./data_stream_control.py --data_file $file_name --search_method BO --Index $2 --search_budget $search_budget --seed 2 --query_type $1

wait

echo "grid_search, budget $search_budget"

# python3 ./data_stream_control.py --data_file $file_name --search_method grid_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
python3 ./data_stream_control.py --data_file $file_name --search_method grid_search --Index $2 --search_budget $search_budget --seed 0 --query_type $1
# python3 ./data_stream_control.py --data_file $file_name --search_method grid_search --Index $2 --search_budget $search_budget --seed 2 --query_type $1


wait


echo "RL eval, budget $search_budget"
# python3 RL_eval_single.py --RL_policy DDPG --Index $2 --load_model default --data_file $file_name --search_budget $search_budget --seed 0 --query_type $1
python3 RL_eval_single.py --RL_policy DDPG --Index $2 --load_model default --data_file $file_name --search_budget $search_budget --seed 0 --query_type $1
# python3 RL_eval_single.py --RL_policy DDPG --Index $2 --load_model default --data_file $file_name --search_budget $search_budget --seed 2 --query_type $1