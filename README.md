This is the code base for our technical report <b>SARA: Salience-Aware Reinforced Adaptive Decoding for Large Language Models in Abstractive Summarization</b>

#### Install required packages without transformers library

```
pip install -r requirements.txt
```

#### Install modified-transformers locally

```
cd transformers
pip install -e .
```

#### Download dataset and weight of LLMs

- Download the processed datasets from [this baidu drive link](https://pan.baidu.com/s/1uALs17MOyS1A_Rk49NN2Nw) extract code: a259
- Download the weight of LLMs (e.g., gpt-neo-2.7B)

#### Run the scripts

For gpt-neo model on cnn\_dailymail dataset:

```
cd src
CUDA_VISIBLE_DEVICES=0 python test_performance_decoder_new_fc.py
	--model_name_or_path ../gpt-neo-2.7B
	--dataset cnn_dailymail_new
	--data_type len10_notriblock2_order
	--do_sample
	--num_samples 100
	--max_input_length 1024
	--loading_mode bf16
	--min_new_tokens 30
	--max_new_tokens 70
	--sample_top_k 100
	--sample_top_p 0.95
	--baseline_top_k 10
	--baseline_top_p 0.5
	--test_top_k 50
	--test_top_p 0.9
	--repetition_penalty 1.0
	--context_aware_decoding_alpha 1.0
	--cad_full 1.0
	--cad_salience 0.0
	--num_beams 1
	--temperature 1.0
	--batch_size 4
	--save_output
	--epoch_num 1
	--losses_log_every 100
	--save_checkpoint_every 100
	--save_checkpoint_path ../output/models/gpt-neo-2.7B_log_cnndm_new_order_sqrt
	--id test
	--save_history_ckpt 0
	--start_from model_output
	--lr 1e-5
	--num_return_sequences 5
	--alpha_et_hidden_size 4096
	--random_seed 42
	--RougeL_reward_weight 1.0
	--Rouge1_reward_weight 1.0
	--Rouge2_reward_weight 1.0
	--factkb_weight 1.0
	--dropout_rate 0.0
	--do_train
	--do_test
	--test_all
	--sqrt_dimension 1
	--logging ../output/logs/gpt-neo-2.7B_log_cnndm_new_order_sqrt.log
```

For Llama-2-7b-chat-hf model on cnn\_dailymail dataset:

```
cd src
CUDA_VISIBLE_DEVICES=0 python test_performance_decoder_new_fc.py 
	--model_name_or_path ../Llama-2-7b-chat-hf 
	--dataset cnn_dailymail_new 
	--data_type len10_notriblock2_order 
	--num_samples 20 
	--max_input_length 1024 
	--loading_mode bf16 
	--min_new_tokens 30 
	--max_new_tokens 70 
	--do_sample 
	--sample_top_k 100 
	--sample_top_p 0.95 
	--baseline_top_k 10 
	--baseline_top_p 0.5 
	--test_top_k 50 
	--test_top_p 0.9 
	--repetition_penalty 1.0 
	--context_aware_decoding_alpha 1.0 
	--cad_full 1.0 
	--cad_salience 0.0 
	--num_beams 1 
	--sample_num_beams 1 
	--temperature 1.0 
	--batch_size 2 
	--save_output 
	--output_fname "" 
	--epoch_num 1 
	--losses_log_every 100 
	--save_checkpoint_every 100 
	--save_checkpoint_path ../output/models/Llama-2-7b-chat-hf_sqrt_concate_bf16 
	--id test 
	--save_history_ckpt 0 
	--start_from model_output 
	--lr 1e-05 
	--num_return_sequences 5 
	--alpha_et_hidden_size 4096 
	--random_seed 42 
	--RougeL_reward_weight 1.0 
	--Rouge1_reward_weight 1.0 
	--Rouge2_reward_weight 1.0 
	--factkb_weight 1.0 
	--max_grad_norm 1.0 
	--dropout_rate 0.2 
	--warmup_step 100 
	--warmup_train_step 2000 
	--accumulation_steps 1 
	--scheduler_type linear 
	--do_test 
	--test_all 
	--sqrt_dimension 1 
	--sqrt_dimension_first 1 
	--sqrt_dimension_second 1 
	--logging ../output/logs/Llama-2-7b-chat-hf_sqrt_concate_bf16.log
```

#### Thanks
This project is modified from https://github.com/zhichaoxu-shufe/context-aware-decoding-qfs.
