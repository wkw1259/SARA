# DEVICE=0

# top_k=50
# top_p=0.9

# # CUDA_VISIBLE_DEVICES=$DEVICE 
# python test_performance_decoder.py \
# --model_name_or_path /home/weikaiwen/na/context-aware-decoding-qfs/data/pretrained_models/gpt-neo-2.7B \
# --top_k $top_k \
# --context_aware_decoding_alpha 0 \
# --cad_full 1 \
# --cad_salience 0 \
# --num_samples 100 \
# --batch_size 4 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname gpt-neo_full1salience0cad15 \
# --dataset cnn_dailymail \
# --do_sample \
# --epoch_num 2 \
# --losses_log_every 25 \
# --save_checkpoint_every 2500 \
# --checkpoint_path model_output \
# --id test \
# --save_history_ckpt 0 \
# --start_from model_output

# python test_performance_decoder.py --model_name_or_path /home/weikaiwen/na/context-aware-decoding-qfs/data/pretrained_models/gpt-neo-2.7B --top_k 50 --context_aware_decoding_alpha 1.0 --cad_full 1 --cad_salience 0 --num_samples 100 --batch_size 2 --max_input_length 512 --min_new_tokens 30 --max_new_tokens 70 --save_output --output_fname gpt-neo_full1salience0cad15 --dataset cnn_dailymail --do_sample --epoch_num 2 --losses_log_every 100 --save_checkpoint_every 2500 --checkpoint_path model_output --id test --save_history_ckpt 0 --start_from model_output

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 1 \
# --cad_salience 0 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience0cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0 \
# --cad_salience 1 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full0salience1cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0.9 \
# --cad_salience 0.1 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience9cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0.8 \
# --cad_salience 0.2 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience9cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0.7 \
# --cad_salience 0.3 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience9cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0.2 \
# --cad_salience 0.8 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience9cad15 \
# --dataset cnn_dailymail \
# --do_sample

# CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
# --model_name_or_path /data/nayu/pretrained_models/Llama-2-7b-chat-hf \
# --top_k $top_k \
# --context_aware_decoding_alpha 0.15 \
# --cad_full 0.1 \
# --cad_salience 0.9 \
# --num_samples 100 \
# --batch_size 16 \
# --max_input_length 512 \
# --min_new_tokens 30 \
# --max_new_tokens 70 \
# --save_output \
# --output_fname llama2_7b_chat_full1salience9cad15 \
# --dataset cnn_dailymail \
# --do_sample


python test_performance_decoder_new_fc.py \
    --model_name_or_path /root/autodl-tmp/code/na/context-aware-decoding-qfs/data/pretrained_models/Llama-2-7b-chat-hf \
    --dataset cnn_dailymail \
    --num_samples 100 \
    --max_input_length 512 \
    --loading_mode fp32 \
    --min_new_tokens 30 \
    --max_new_tokens 70 \
    --do_sample \
    --sample_top_k 50 \
    --sample_top_p 0.9 \
    --baseline_top_k 10 \
    --baseline_top_p 0.5 \
    --test_top_k 50 \
    --test_top_p 0.9 \
    --repetition_penalty 1.0 \
    --context_aware_decoding_alpha 1.0 \
    --cad_full 1.0 \
    --cad_salience 0.0 \
    --num_beams 1 \
    --temperature 1.0 \
    --batch_size 2 \
    --save_output \
    --epoch_num 1 \
    --losses_log_every 100 \
    --save_checkpoint_every 100 \
    --checkpoint_path ../output/models/Llama-2-7b-chat-hf_model_output_training_all_test100step_lr1e5_bs2_return5_hidden4096_weight101010_warmup500to2000_newfc_nologsoft_dropout03_topk5010 \
    --id test \
    --save_history_ckpt 0 \
    --start_from model_output \
    --lr 1e-5 \
    --num_return_sequences 5 \
    --alpha_et_hidden_size 4096 \
    --random_seed 42 \
    --RougeL_reward_weight 1.0 \
    --Rouge1_reward_weight 1.0 \
    --Rouge2_reward_weight 1.0 \
    --factkb_weight 0.0 \
    --dropout_rate 0.3 \
    --logging ../output/logs/Llama-2-7b-chat-hf_model_output_training_all_test100step_lr1e5_bs2_return5_hidden4096_weight101010_warmup500to2000_newfc_nologsoft_dropout03_topk5010.log

# # 100



python test_performance_decoder_only_save_fc.py \
    --model_name_or_path /root/autodl-tmp/code/na/context-aware-decoding-qfs/data/pretrained_models/opt-13b \
    --dataset cnn_dailymail \
    --num_samples 100 \
    --max_input_length 512 \
    --loading_mode fp32 \
    --min_new_tokens 30 \
    --max_new_tokens 70 \
    --do_sample \
    --top_k 50 \
    --top_p 0.9 \
    --repetition_penalty 1.0 \
    --context_aware_decoding_alpha 1.0 \
    --cad_full 1.0 \
    --cad_salience 0.0 \
    --num_beams 1 \
    --temperature 1.0 \
    --batch_size 2 \
    --save_output \
    --epoch_num 1 \
    --losses_log_every 100 \
    --save_checkpoint_every 100 \
    --checkpoint_path ../output/models/opt-13b_model_output_training_all_test100step_lr1e5_bs2_return2_hidden200_weight101010 \
    --id test \
    --save_history_ckpt 0 \
    --start_from model_output \
    --lr 1e-5 \
    --num_return_sequences 2 \
    --alpha_et_hidden_size 200 \
    --random_seed 42 \
    --RougeL_reward_weight 1.0 \
    --Rouge1_reward_weight 1.0 \
    --Rouge2_reward_weight 1.0 \
    --factkb_weight 0.0 \
    --logging ../output/logs/opt-13b_model_output_training_all_test100step_lr1e5_bs2_return2_hidden200_weight101010.log