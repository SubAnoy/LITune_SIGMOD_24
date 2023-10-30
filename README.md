# LITune SIGMOD 2024 Setup and Evaluation Guide


This is code repo for Paper LITune: Tuning Learned Index Using Deep Reinforcement Learning

Track Name: Research Paper 4th Submission Round

Paper ID: 827


## 1. Prerequisites:
* Ensure you have `g++` installed with support for `C++17` and the required flags.
* Python environment set up, preferably using a virtual environment for isolation.

## 2. Dependicies on External Learned Index Repo

> ALEX: [GitHub Repository](https://github.com/microsoft/ALEX)

> CARMI: [Research Paper](https://www.vldb.org/pvldb/vol15/p2679-gao.pdf)


## 3. Running Environment:

Firstly, make sure you have the required C++ compile environment set up. It should support both ALEX and PGM use cases. Run the following commands for compilation:

```
g++ ./Index/Alex/index_test.cpp -w -std=c++17 -msse4 -mpopcnt -mlzcnt -mbmi -o ./Index/Alex/exe_alex_index
./Index/Alex/exe_alex_index ./data_SOSD/{data_file_name} {query_type}
```

Here we provided the dafault sample data chunks stored in `data_SOSD` folder from SOSD OSM (For SOSD, see here `https://github.com/learnedsystems/SOSD`). We prepare four datasets in total: *books*, *fb*, *OSM*, *MIX*. Others are stored in `./data_SOSD_other`

For Python dependencies, you can use `pip` to install the required libraries:

- Set up a virtual environment (optional but recommended):

```
python3 -m venv litune_venv
source litune_venv/bin/activate
```

- Install the Python requirements:

```
pip install -r requirements.txt
```

## 4. Evaluation:

For a more extensive list of options and flexible configurations/arguments, delve into the Python controller files located in the ./scripts directory or check the provided shell execution scripts.

### Evaluation Pointers:
> - Note: Please be cautious when using CARMI for certain use cases, as it may exhaust system memory and potentially cause system instability or crashes. Always ensure you have adequate backups and monitor resource usage during evaluations.
> 
> - Here, we just take ALEX as an example, please use the similar command for testing on CARMI
>
> - For all evaluations, repeat with `query_type` as `read-heavy` and `write-heavy` as mentioned
> - Sample queries and workloads can be found in `./Index/Alex/index_test.cpp` or `./Index/CARMI/index_test.cpp`


**Pre-training LITune-RL**:

> You have two options available:

> - You can train the model from scratch using the provided scripts.
> - For a quicker approach, you can directly proceed to the evaluations section by utilizing the pre-trained model located in `./rlmodels`.


- Pre-training the RL model for ALEX:

```
sh exe_RL_training.sh
```

**Static Experiments**:


- Step 1: Run the evaluation for different budgets and output results:

```
sh eval_single_budget.sh balanced ALEX | tee ./logs/single_budget_balanced.txt
```

- Step 2: Run evaluations with rich budget:


```
sh exe.sh balanced ALEX | tee ./logs/full_budget_balanced.txt
```

**Data Shifts**:

- Step 1: Run the evaluations without the O2 system:

```
sh exe_stream.sh balanced ALEX | tee ./logs/stream_balanced.txt
```

- Step 2: Run the evaluations with the O2 system:

```
sh exe_O2.sh
```




