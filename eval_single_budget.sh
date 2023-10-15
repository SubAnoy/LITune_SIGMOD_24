echo "Start Testing Data with budget......"

if [ -z "$3" ]; then
    arg_budget_adds_on=0
else
    arg_budget_adds_on=$3
fi

file_name=data_0
search_budget=$((50 + $arg_budget_adds_on))

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

search_budget=$((100 + $arg_budget_adds_on))

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


search_budget=$((150 + $arg_budget_adds_on))

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


search_budget=$((200 + $arg_budget_adds_on))

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


search_budget=$((250 + $arg_budget_adds_on))

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

echo "End Testing......"
