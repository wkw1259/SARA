import os
import sys
import time
import json
import logging
import argparse
import math
import random

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup 


import torch
import torch.nn.functional as F

import evaluate

from utils import *

from collections import defaultdict

'''
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
'''



def save_checkpoint(opt, model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.save_checkpoint_path):
        os.makedirs(opt.save_checkpoint_path)

    # 只存fc层
    fc_layers_state_dict = {
    'my_all_f': model.my_all_f.state_dict(),
    'my_all_f1': model.my_all_f1.state_dict(),
    'my_f': model.my_f.state_dict(),
    }

    save_checkpoint_path = os.path.join(opt.save_checkpoint_path, 'model%s_fc_layers.pth' %(append))
    torch.save(fc_layers_state_dict, save_checkpoint_path)
    logger.info("model saved to {}".format(save_checkpoint_path))
    optimizer_path = os.path.join(opt.save_checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)
    # with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
    #     pickle_dump(infos, f)
    # if histories:
    #     with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
    #         pickle_dump(histories, f)
            
def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, tokenizer, RougeL_reward_weight, Rouge1_reward_weight, 
                             Rouge2_reward_weight, factkb_weight, logger, batch_input):
    
    batch_size = len(data_gts)  # 2
    gen_result_size = gen_result.shape[0]  # 4
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img    2
    assert greedy_res.shape[0] == batch_size

    documents = []  # 用于factKB

    res = OrderedDict()
    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    # data_gts = data_gts.data.cpu().numpy()
    
    # 把topk和greedy的生成结果都放到res中
    for i in range(gen_result_size):  # 4
        res[i] = [gen_result[i]]
    for i in range(batch_size):
        res[gen_result_size + i] = [greedy_res[i]]  # 4 topk + 2 greedy 

    gts = OrderedDict()
    for i in range(len(data_gts)):
        # gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
        gts[i] = [data_gts[i]]
        documents.append(batch_input[i]) # batch_input代表输入的长context

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})   # 4 topk + 2 greedy的真值
    
    documents_ = [documents[i // seq_per_img] for i in range(gen_result_size)]
    documents_.extend(documents[i] for i in range(batch_size))  # 每条input和生成的结果对齐
    
    res_for_evaluate, gt_for_evaluate = [], []
    
    for k, v in res__.items():
        res_for_evaluate.append(tokenizer.decode(v[0].tolist(), skip_special_tokens=True, reduce_tokenization_space=True))
    
    for k, v in gts_.items():
        gt_for_evaluate.append(tokenizer.decode(v[0].tolist(), skip_special_tokens=True, reduce_tokenization_space=True))
    
    # if cider_reward_weight > 0:
    #     _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
    #     print('Cider scores:', _)
    # else:
    #     cider_scores = 0
    # if bleu_reward_weight > 0:
    #     _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
    #     bleu_scores = np.array(bleu_scores[3])
    #     print('Bleu scores:', _[3])
    # else:
    #     bleu_scores = 0
    
    # 初始化一个列表来存储每对序列的RougeLsum、Rouge1、Rouge2
    scores, Rouge1_scores, Rouge2_scores, Factkb_scores = [], [], [], []
    evaluator = Evaluator()
    
    # 逐条计算rougeLsum_fmeasure分数
    for pred, gt, document in zip(res_for_evaluate, gt_for_evaluate, documents_):
        pred = [pred]
        gt = [gt]
        document = [document]
        result_dict = evaluator.evaluate(pred, gt, document, metrics=["rouge", 'factkb'])
        scores.append(result_dict['rougeLsum_fmeasure'])
        Rouge1_scores.append(result_dict['rouge1_fmeasure'])
        Rouge2_scores.append(result_dict['rouge2_fmeasure'])
        Factkb_scores.append(result_dict['factkb'])

    # for k, v in result_dict.items():
    #     print(f"{k} -> {v*100:.2f}")
    # for k, v in rouge_dict.items():
    #     result_dict[k] = v

    scores, Rouge1_scores, Rouge2_scores, Factkb_scores = np.array(scores), np.array(Rouge1_scores), np.array(Rouge2_scores), np.array(Factkb_scores)

    # 打印第一个样本的 topk 和 greedy结果看看
    topk_scores, topk_Rouge1_scores, topk_Rouge2_scores, topk_factkb_scores = [], [], [], []
    for sample_i in range(seq_per_img):
        # print('sample_i', sample_i)
        # print('seq_per_img', seq_per_img)
        topk1_score, topk1_Rouge1_score, topk1_Rouge2_scores, topk1_factkb_scores = scores[sample_i], Rouge1_scores[sample_i], Rouge2_scores[sample_i], Factkb_scores[sample_i]
        logger.info('first sample {}-th topk rougeL F: {:.3f}, topk rouge1 F: {:.3f}, topk rouge2 F: {:.3f}, Factkb_scores: {:.3f}'.format(str(sample_i), topk1_score, topk1_Rouge1_score, topk1_Rouge2_scores, topk1_factkb_scores))
        print('first sample {}-th topk rougeL F: {:.3f}, topk rouge1 F: {:.3f}, topk rouge2 F: {:.3f}, Factkb_scores: {:.3f}'.format(str(sample_i), topk1_score, topk1_Rouge1_score, topk1_Rouge2_scores, topk1_factkb_scores))
        topk_scores.append(topk1_score)
        topk_Rouge1_scores.append(topk1_Rouge1_score)
        topk_Rouge2_scores.append(topk1_Rouge2_scores)
        topk_factkb_scores.append(topk1_factkb_scores)
        
    logger.info('avg topk rougeL F: {:.3f}, topk rouge1 F: {:.3f}, topk rouge2 F: {:.3f}, Factkb_scores: {:.3f}'.format(sum(topk_scores) / len(topk_scores), 
                                                                                                 sum(topk_Rouge1_scores) / len(topk_Rouge1_scores), 
                                                                                                 sum(topk_Rouge2_scores) / len(topk_Rouge2_scores),
                                                                                                 sum(topk_factkb_scores) / len(topk_factkb_scores),
                                                                                                 ))

    greedy1_score, greedy1_Rouge1_score, greedy1_Rouge2_scores, greedy1_factkb_scores = scores[-batch_size], Rouge1_scores[-batch_size], Rouge2_scores[-batch_size], Factkb_scores[-batch_size]
    logger.info('first sample greedy rougeL F: {:.3f}, greedy rouge1 F: {:.3f}, greedy rouge2 F: {:.3f}, greedy factkb: {:.3f}'.format(greedy1_score, greedy1_Rouge1_score, greedy1_Rouge2_scores, greedy1_factkb_scores))

    # 针对不同的指标，分别计算topk和greedy的差值
    # scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]  # （2,2）topk - （2,1）greedy
    # scores = scores.reshape(gen_result_size)

    # Rouge1_scores = Rouge1_scores[:gen_result_size].reshape(batch_size, seq_per_img) - Rouge1_scores[-batch_size:][:, np.newaxis]  # （2,2）topk - （2,1）greedy
    # Rouge1_scores = Rouge1_scores.reshape(gen_result_size)

    # Rouge2_scores = Rouge2_scores[:gen_result_size].reshape(batch_size, seq_per_img) - Rouge2_scores[-batch_size:][:, np.newaxis]  # （2,2）topk - （2,1）greedy
    # Rouge2_scores = Rouge2_scores.reshape(gen_result_size)

    # Factkb_scores = Factkb_scores[:gen_result_size].reshape(batch_size, seq_per_img) - Factkb_scores[-batch_size:][:, np.newaxis]  # （2,2）topk - （2,1）greedy
    # Factkb_scores = Factkb_scores.reshape(gen_result_size)

    # scores = RougeL_reward_weight * scores + Rouge1_reward_weight * Rouge1_scores + Rouge2_reward_weight * Rouge2_scores + factkb_weight * Factkb_scores

    # TODO 和self-critical的reward计算方式一致
    scores = RougeL_reward_weight * scores + Rouge1_reward_weight * Rouge1_scores + Rouge2_reward_weight * Rouge2_scores + factkb_weight * Factkb_scores
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    
    # scores_tmp = RougeL_reward_weight * scores + Rouge1_reward_weight * Rouge1_scores + Rouge2_reward_weight * Rouge2_scores + factkb_weight * Factkb_scores
    # scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores
    # print('scores', scores)
    # print('Rouge1_scores', Rouge1_scores)
    # print('Rouge2_scores', Rouge2_scores)
    # print('Factkb_scores', Factkb_scores)

    # logger.info('weighted sum scores {}', scores)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N,L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq>0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output


def test(args, test_set, logger, tokenizer, DEVICE, model, generation_config):
    # print('test_generation_config', generation_config)
    dataset = test_set
    num_batch = len(dataset)//args.test_batch_size+1 if len(dataset)%args.test_batch_size!=0 else len(dataset)//args.test_batch_size

    # if args.context_aware_decoding_alpha > 0.:
    #     null_input = get_null_input_decoder(args.dataset)

    logger.info("start decoding!")
    predictions, references, documents = [], [], []
    start_time = time.time()
    total_decoding_length = 0
    for batch_idx, _ in tqdm(enumerate(range(num_batch))):
        batch = dataset[args.test_batch_size*batch_idx: args.test_batch_size*(batch_idx+1)]

        batch_input, batch_reference, batch_presumm = [row[0] for row in batch], [row[1] for row in batch], [row[2] for row in batch]
        tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=1800, padding=True, truncation=True)
        if args.context_aware_decoding_alpha >=0.: #full and salience and prompt
            print('full+salience-prompt generation')
            batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
            tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", max_length=1800, padding=True, truncation=True) 
            batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
            tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", max_length=1800, padding=True, truncation=True)
            with torch.no_grad():
                #model.train()
                # frozen llm, learn linear
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                    null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                )
                
        elif args.cad_salience >= 0.: #full and salience
            print('full+salience generation')
            batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
            tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", max_length=1800, padding=True, truncation=True)
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                ) 
        else:
            print('standard generation')
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                )
        predictions.extend(tokenizer.batch_decode(output[:, tokenized_input.input_ids.shape[1]:], skip_special_tokens=True, reduce_tokenization_space=True))
        references.extend(batch_reference)
        total_decoding_length += output[:, tokenized_input.input_ids.shape[1]:].shape[1]
        documents.extend(batch_input)

        ## 当训练稳定下来了打印显存大小
        if batch_idx == 5:
            # from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
            #     nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown
            # try:
            #     nvmlInit()
            #     device_count = nvmlDeviceGetCount()
            #     # for i in range(device_count):
            #     i = 1
            #     handle = nvmlDeviceGetHandleByIndex(i)
            #     # name = nvmlDeviceGetName(handle)
            #     memory_info = nvmlDeviceGetMemoryInfo(handle)
            #     total_memory = memory_info.total / (1024 ** 2)  # MB
            #     used_memory = memory_info.used / (1024 ** 2)    # MB
            #     free_memory = memory_info.free / (1024 ** 2)    # MB
                
            #     # print(f"GPU {i}: {name}")
            #     logger.info(f"  总显存: {total_memory:.2f} MB")
            #     logger.info(f"  已用显存: {used_memory:.2f} MB")
            #     logger.info(f"  可用显存: {free_memory:.2f} MB")
            #     logger.info("-" * 30)
            # except Exception as e:
            #     logger.info(f"获取显存信息时出错: {e}")
            # finally:
            #     nvmlShutdown()

            def get_gpu_memory():
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
                    memory_cached = torch.cuda.memory_reserved() / 1024**2  # 转换为MB
                    logger.info(f"当前PyTorch显存使用: {memory_allocated:.2f} MB")
                    logger.info(f"当前PyTorch显存缓存: {memory_cached:.2f} MB")
                else:
                    logger.info("未检测到可用GPU")
                    
            get_gpu_memory()
        # break
    
    print(f"total decoding takes {time.time()-start_time:.2f} seconds!")
    print(f"average token per seconds -> {(time.time()-start_time)/total_decoding_length:.5f}")

    # del model  # offload model to avoid memory spike

    assert len(predictions)==len(references), "mismatched shape, force exit!"
    evaluator = Evaluator()
    result_dict = evaluator.evaluate(predictions, references, documents, metrics=["rouge", "sacre_bleu", "bertscore", "factkb"])
    for k, v in result_dict.items():
        logger.info(f"{k} -> {v*100:.2f}")
    logger.info("\n")

    # if args.save_output:
        # if not args.output_fname:
        #     model_name = config._name_or_path.split("/")[-1]
        #     args.output_fname = f"{model_name}_max_new_tokens_{args.max_new_tokens}_topk_{args.top_k}_topp_{args.top_p}_alpha_{args.context_aware_decoding_alpha}.jsonl"
        #     args.output_fname = os.path.join("./generation", args.output_fname)

        # with open(args.output_fname, "w") as fout:
        #     for i in range(len(predictions)):
        #         json_line = {"prediction": predictions[i], "reference:": references[i]}
        #         json.dump(json_line, fout)
        #         fout.write("\n")
        # fout.close()
        # logger.info(f"generation file -> {args.output_fname}\n")

    # scores.append(result_dict['rougeLsum_fmeasure'])
    # Rouge1_scores.append(result_dict['rouge1_fmeasure'])
    # Rouge2_scores.append(result_dict['rouge2_fmeasure'])
    # Factkb_scores.append(result_dict['factkb'])
    
    return (result_dict['rougeLsum_fmeasure'], result_dict['rouge1_fmeasure'], result_dict['rouge2_fmeasure'], result_dict['factkb'])   # 返回rouge-L


def convert_my_layers_to_fp32(model):
    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        # 如果模块的名字以 'my_' 开头，转换为 fp32
        if name.startswith('my_'):
            # 使用 .to(torch.float32) 转换为 fp32
            for param in module.parameters():
                param.data = param.data.to(torch.float32)  # 转换权重
            if isinstance(module, nn.Module):
                module = module.to(torch.float32)  # 转换模块的其他部分（如偏置等）
            print(f"Converted {name} to fp32")
    return model


if __name__ == "__main__":
    
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--data_type", type=str, default="len10_notriblock")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--loading_mode", type=str, default="fp32")
    parser.add_argument("--min_new_tokens", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--sample_top_p", type=float, default=0.9)
    parser.add_argument("--baseline_top_k", type=int, default=50)
    parser.add_argument("--baseline_top_p", type=float, default=0.9)
    parser.add_argument("--test_top_k", type=int, default=50)
    parser.add_argument("--test_top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    parser.add_argument("--cad_full", type=float, default=1.0)
    parser.add_argument("--cad_salience", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--sample_num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_fname", type=str, default="")
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')  
    parser.add_argument('--save_checkpoint_every', type=int, default=25,  # 2500
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--save_checkpoint_path', type=str, default=None,
                    help='directory to store checkpointed models')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                    help='directory to store checkpointed models')
    parser.add_argument('--load_best', type=int, default=1, help='whether load best ckpt during testing')
    parser.add_argument('--load_ckpt_num', type=str, default=None,
                    help='load ckpt num')

    parser.add_argument('--id', type=str, default='test',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--save_history_ckpt', type=int, default=0,
                    help='If save checkpoints at every save point')
    parser.add_argument('--start_from', type=str, default='model_output',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'model.pth'         : weights
                    """)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--alpha_et_hidden_size', type=int, default=4096)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--RougeL_reward_weight', type=float, default=1.0)
    parser.add_argument('--Rouge1_reward_weight', type=float, default=0.0)
    parser.add_argument('--Rouge2_reward_weight', type=float, default=0.0)
    parser.add_argument('--factkb_weight', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--debug_flag", action="store_true", help='whether debug on a small portion of data')
    parser.add_argument("--debug_num", type=int, default=5)
    parser.add_argument("--use_log_softmax", action="store_true", help='whether to use log_softmax')
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--warmup_step", type=float, default=100)
    parser.add_argument("--warmup_train_step", type=float, default=2000)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--random_train_set", action="store_true", help='whether shuffle train set')
    parser.add_argument("--scheduler_type", type=str, default="linear", help='linear or cosine')
    parser.add_argument("--do_train", action="store_true", help='whether do training')
    parser.add_argument("--do_test", action="store_true", help='whether do testing')

    parser.add_argument("--test_all", action="store_true", help='whether do testing on all dataset')
    parser.add_argument("--save_every_log_step", action="store_true", help='whether save all the ckpt at logging time')
    parser.add_argument("--sqrt_dimension", type=int, default=1, help='whether do sqrt dimension on hidden states')  # todo 最后记得改成store_true
    parser.add_argument("--sqrt_dimension_first", type=int, default=1, help='whether do sqrt dimension on hidden states')  # todo 最后记得改成store_true
    parser.add_argument("--sqrt_dimension_second", type=int, default=1, help='whether do sqrt dimension on hidden states')  # todo 最后记得改成store_true
    parser.add_argument("--sqrt_method", type=str, default='concate_dim', help='concate_dim/ hidden_dim / log / sqrt05 / head_dim') 
    parser.add_argument("--my_fc_fp32", action="store_true", help='whether let my fc layer fp32')
    parser.add_argument("--ablation_main_sequence", action="store_true", help='whether ablation main sequence, alpha=0')
    parser.add_argument("--ablation_presumm_sequence", action="store_true", help='whether ablation presumm sequence, beta=0')
    parser.add_argument("--ablation_null_sequence", action="store_true", help='whether ablation null sequence, gamma=0')
    parser.add_argument("--test_factkb_weight", type=float, default=1.0, help='weight of factkb during model selection')
    parser.add_argument("--logging", type=str, default="./default_decoder.log")
    args = parser.parse_args()

    # 打印全部
    # torch.set_printoptions(threshold=torch.inf)

    # logging.basicConfig(
    #     format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #     datefmt='%m/%d/%Y %H:%M:%S',
    #     level=logging.INFO,
    #     filename=args.logging, 
    #     filemode='a',
    #     )
    # logger = logging.getLogger(__name__)

    # 创建一个日志记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录的最低级别
    # 创建一个控制台处理器，并设置级别为INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # 创建一个文件处理器，并设置级别为DEBUG
    file_handler = logging.FileHandler(args.logging)
    file_handler.setLevel(logging.DEBUG)
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    for k, v in vars(args).items():
        logger.info(f"{k} -> {v}")
    logger.info(f"\n")
    
    # # 设置随机种子
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

    train_set, validation_set, test_set = load_dataset(args.dataset, args.data_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    if args.debug_flag:
        train_set = pretokenize(train_set[:args.debug_num], tokenizer, args.max_input_length)
        all_test_set = pretokenize(test_set[0:args.debug_num], tokenizer, args.max_input_length)
        test_set = pretokenize(test_set[:args.debug_num], tokenizer, args.max_input_length)
        args.losses_log_every = 2
        args.save_checkpoint_every = 2
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)
    else:
        if args.random_train_set:
            random.shuffle(train_set)
        train_set = pretokenize(train_set, tokenizer, args.max_input_length)
        if args.test_all:  # 完整的test set
            all_test_set = pretokenize(test_set, tokenizer, args.max_input_length)
        else:
            all_test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)
        test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)

    train_set = [[template_input_decoder(row, args.dataset), row[1], row[2]] for row in train_set]
    test_set = [[template_input_decoder(row, args.dataset), row[1], row[2]] for row in test_set]
    all_test_set = [[template_input_decoder(row, args.dataset), row[1], row[2]] for row in all_test_set]
    
    print('loading model checkpoint')
    model = configure_model_loading(args)
    if args.load_checkpoint_path:
        if args.load_best:
            logger.info('Load pretrained weights at {}'.format(os.path.join(args.load_checkpoint_path, 'model-best.pth')))
            fc_path = os.path.join(args.load_checkpoint_path, 'model-%s_fc_layers.pth' %('best'))
            fc_layers_state_dict = torch.load(fc_path)
            model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
            model.my_all_f1.load_state_dict(fc_layers_state_dict['my_all_f1'])
            model.my_f.load_state_dict(fc_layers_state_dict['my_f'])
        else:
            logger.info('Load pretrained weights at {}'.format(os.path.join(args.load_checkpoint_path, 'model-%s_fc_layers.pth' %('1100'))))
            fc_path = os.path.join(args.load_checkpoint_path, 'model-%s_fc_layers.pth' %('1100'))  # 这里要改
            fc_layers_state_dict = torch.load(fc_path)
            model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
            model.my_all_f1.load_state_dict(fc_layers_state_dict['my_all_f1'])
            model.my_f.load_state_dict(fc_layers_state_dict['my_f'])
    
    # 是否将fc层转换为fp32
    if args.my_fc_fp32:
        model = convert_my_layers_to_fp32(model)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 冻结除了最后两个全连接层之外的所有参数
    for name, param in model.named_parameters():
        # 最后两个全连接层的名称包含 'my_'
        if 'my_' in name:
            print(f"Training {name}")
        else:
            # print(f"Freezing {name}")
            param.requires_grad = False
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"训练层: {name}")

    # 将模型移动到 GPU
    model = model.to(DEVICE)
    
    sample_generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.sample_num_beams,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        cad_full=args.cad_full,
        cad_salience=args.cad_salience,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ablation_main_sequence=args.ablation_main_sequence,
        ablation_presumm_sequence=args.ablation_presumm_sequence,
        ablation_null_sequence=args.ablation_null_sequence,
    )

    baseline_generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.baseline_top_k,
        top_p=args.baseline_top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        cad_full=args.cad_full,
        cad_salience=args.cad_salience,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ablation_main_sequence=args.ablation_main_sequence,
        ablation_presumm_sequence=args.ablation_presumm_sequence,
        ablation_null_sequence=args.ablation_null_sequence,
    )

    test_generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.test_top_k,
        top_p=args.test_top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        cad_full=args.cad_full,
        cad_salience=args.cad_salience,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ablation_main_sequence=args.ablation_main_sequence,
        ablation_presumm_sequence=args.ablation_presumm_sequence,
        ablation_null_sequence=args.ablation_null_sequence,
    )


    ###################
    ####   train   ####
    ###################
    if args.do_train:
        # 用来存历史数据
        histories = defaultdict(dict)
        infos = {
            'iter': 0,
            # 'epoch': 0,
            # 'loader_state_dict': None,
            # 'vocab': loader.get_vocab(),
        }
        iteration = infos['iter']
        best_val_score = None
        
        logger.info('prepare training set')
        dataset = train_set  #   test_set  train_set
        num_batch = len(dataset)//args.batch_size+1 if len(dataset)%args.batch_size!=0 else len(dataset)//args.batch_size

        print('num_batch', num_batch)

        # if args.context_aware_decoding_alpha > 0.:
        #     null_input = get_null_input_decoder(args.dataset)
        
        #  Build optimizer
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
        
        # 创建学习率调度器，设置 warmup 的步数
        num_training_steps = args.epoch_num * num_batch
        if args.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
            num_warmup_steps=args.warmup_step,   # int(num_training_steps//10),  # warmup步数取得总步数的10%
                num_training_steps=args.warmup_train_step # num_training_steps
            )
        elif args.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_step,   # int(num_training_steps//10),  # warmup步数取得总步数的10%
                num_training_steps=args.warmup_train_step # num_training_steps
            )

        logger.info("start training!")
        
        for epoch_i in range(args.epoch_num):
            # predictions, references, documents = [], [], []
            start = time.time()
            total_decoding_length = 0
            for batch_idx, _ in tqdm(enumerate(range(num_batch))):
                batch = dataset[args.batch_size*batch_idx: args.batch_size*(batch_idx+1)]

                batch_input, batch_reference, batch_presumm = [row[0] for row in batch], [row[1] for row in batch], [row[2] for row in batch]
                tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=1800, padding=True, truncation=True)
                if args.context_aware_decoding_alpha >=0.: #full and salience and prompt
                    # print('full+salience-prompt generation')
                    batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
                    tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", max_length=1800, padding=True, truncation=True) 
                    batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
                    tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", max_length=1800, padding=True, truncation=True)
                    
                    tokenized_gt = tokenizer(batch_reference, return_tensors="pt", max_length=1800, padding=True, truncation=True)
                    
                    print('run greedy generate')
                    model.eval()
                    # with torch.no_grad():
                    output_dict = model.generate(
                        input_ids=tokenized_input.input_ids.to(DEVICE),
                        presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                        null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                        attention_mask=tokenized_input.attention_mask.to(DEVICE),
                        presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                        generation_config=baseline_generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )   # output [bs 生成的seq]   scores 生成的seq * [16, 50257]  
                    output_eval = output_dict.sequences
                    greedy_res = output_eval[:, tokenized_input.input_ids.shape[1]:]  # 用生成之后的seq来做损失

                    print('run baseline generate')
                    # 生成多个topk
                    model.train()
                    output_dict = model.generate(
                        input_ids=tokenized_input.input_ids.to(DEVICE),
                        presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                        null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                        attention_mask=tokenized_input.attention_mask.to(DEVICE),
                        presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                        generation_config=sample_generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                    
                    output = output_dict.sequences
                    gen_result, scores = output[:, tokenized_input.input_ids.shape[1]:], output_dict.scores   # scores 70 * [4 * 320000]
                    scores = torch.stack(scores, dim=0).permute(1, 0, 2)  # [bs*num_return_sequences 生成的seq 50257]  
                    if args.use_log_softmax:
                        sample_logprobs = F.log_softmax(scores, dim=-1)
                    else:
                        sample_logprobs = F.softmax(scores, dim=-1)
                    
                    reward = get_self_critical_reward(greedy_res, tokenized_gt.input_ids.to(DEVICE), gen_result, tokenizer, 
                                                    args.RougeL_reward_weight, args.Rouge1_reward_weight, args.Rouge2_reward_weight, args.factkb_weight,
                                                    logger, batch_input)
                    
                    reward = torch.from_numpy(reward).to(sample_logprobs)   
                    rl_crit = RewardCriterion()
                    loss = rl_crit(sample_logprobs, gen_result.data, reward, reduction='mean')

                    criterion = torch.nn.CrossEntropyLoss() # ignore_index=ignore_index
                    # 计算每个生成序列的交叉熵损失
                    # all_losses = []
                    # for i in range(args.num_return_sequences):
                    # i = 0 
                    # generated_logits = scores[:, i, :, :]  # 获取第i个序列的logits (batch_size, sequence_length, vocab_size)
                    # shifted_logits = generated_logits[..., :-1, :].contiguous().view(-1, generated_logits.size(-1))
                    # shifted_labels = tokenized_gt.input_ids[..., 1:].contiguous().view(-1)
                    # CE_loss = criterion(shifted_logits, shifted_labels)
                    # all_losses.append(loss.item())

                    # 打印每个生成序列的交叉熵损失
                    # for i, loss_value in enumerate(all_losses):
                    #     logger.info(f"Cross-entropy loss for sequence {i+1}: {loss_value}")

                    # 计算交叉熵损失，看训练上了没有
                    # Reshape logits (batch_size, num_return_sequences, seq_len, vocab_size)
                    # logits = scores.reshape(tokenized_input.input_ids.shape[0], args.num_return_sequences, scores.shape[1], scores.shape[2])
                    # # 选择每个样本中最高得分的序列以计算损失 (这里我们取第一个序列，可以调整)
                    # logits = logits[:, 0, :, :]  # (batch_size, seq_len, vocab_size)
                    # # Ground truth labels (batch_size, seq_len)
                    # labels = tokenized_gt.input_ids
                    # logits = logits[..., :-1, :].contiguous()
                    # labels = labels[..., 1:].contiguous()

                    # # 获取batch_size和sequence长度
                    # batch_size = logits.shape[0]
                    # sequence_length = min(logits.shape[1], labels.shape[1])  # labels.shape[1]是原始序列的长度
                    # # 确保labels的维度与logits对齐
                    # labels = labels[:, :sequence_length]  # 截断labels
                    # logits = logits[:, :sequence_length, :]  # 截断logits
                    
                    # 设置忽略标签，将填充部分标记为ignore_index
                    # ignore_index = -100
                    # labels = labels.masked_fill(labels == tokenizer.pad_token_id, ignore_index)
                    # logits_mask = labels != ignore_index  # 找到有效的labels
                    # logits = logits.masked_fill(~logits_mask.unsqueeze(-1), float('-inf'))  # 将无效部分设置为 -inf
                    # 将logits和labels展平成二维，以便交叉熵损失计算
                    # logits = logits.reshape(-1, logits.size(-1)).to(DEVICE)
                    # labels = labels.reshape(-1).to(DEVICE)
                    # # 计算交叉熵损失
                    # criterion = torch.nn.CrossEntropyLoss() # ignore_index=ignore_index
                    # CE_loss = criterion(logits, labels)

                    torch.cuda.synchronize()
                    end = time.time()
                    
                    # logger.info("iter {} (epoch {}), avg_loss = {:.3f}, avg_reward = {:.3f}, time/batch = {:.3f}" \
                    #         .format(iteration, epoch_i, CE_loss.item(), loss.item(), end - start))

                    logger.info("iter {} (epoch {}),  avg_reward = {:.3f}, time/batch = {:.3f}" \
                            .format(iteration, epoch_i, loss.item(), end - start))

                    train_loss = loss.item()
                    if math.isinf(train_loss):
                        logger.info("loss == inf, continue")
                        continue
                    
                    optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # if opt.grad_clip_value != 0:
                    #     getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
                    
                    # 梯度累积
                    if iteration % args.accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                    
                    # 如果是最后一个批次时，确保更新一次
                    if (iteration + 1) == len(dataset)//args.batch_size:
                        optimizer.step()
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()  # 获取当前学习率
                        logger.info('Current learning rate: {}'.format(current_lr))

                    # Update the iteration
                    iteration += 1

                    # Write the training loss summary
                    if (iteration % args.losses_log_every == 0):
                        logger.info('loss_history: iteration: {} reward: {}'.format(iteration, loss))

                    # update infos
                    infos['iter'] = iteration

                # elif args.cad_salience >= 0.: #full and salience
                #     print('full+salience generation')
                #     batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
                #     tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", padding=True)
                #     with torch.no_grad():
                #         output = model.generate(
                #             input_ids=tokenized_input.input_ids.to(DEVICE),
                #             presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                #             attention_mask=tokenized_input.attention_mask.to(DEVICE),
                #             presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                #             generation_config=generation_config,
                #         ) 
                # else:
                #     print('standard generation')
                #     with torch.no_grad():
                #         output = model.generate(
                #             input_ids=tokenized_input.input_ids.to(DEVICE),
                #             attention_mask=tokenized_input.attention_mask.to(DEVICE),
                #             generation_config=generation_config,
                #         )
                # predictions.extend(tokenizer.batch_decode(output[:, tokenized_input.input_ids.shape[1]:], skip_special_tokens=True, reduce_tokenization_space=True))
                # references.extend(batch_reference)
                # total_decoding_length += output[:, tokenized_input.input_ids.shape[1]:].shape[1]
                # documents.extend(batch_input)

                # assert len(predictions)==len(references), "mismatched shape, force exit!"
                
                # 每save_checkpoint_every个iteration测一次，并保存比较好的
                if iteration % args.save_checkpoint_every == 0 :
                    ###################
                    ####   test   ####
                    ###################
                    results = test(args, test_set, logger, tokenizer, DEVICE, model, test_generation_config)

                    # rougle1 2 L和factkb 全部加到一起的结果，选加起来最高的ckpt
                    added_results = results[0] + results[1] + results[2] + args.test_factkb_weight * results[3]
                    
                    # 重置存储的结果
                    # predictions, references, documents = [], [], []
                    
                    # Write validation result into summary
                    # histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
                    # logger.info('evaluate rougeL: {}'.format(rougeL_score))

                    logger.info('evaluate added_results: {}'.format(added_results))
                    
                    # Save model if is improving on validation result
                    current_score = added_results

                    best_flag = False

                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    # Dump miscalleous informations
                    logger.info('best_val_score: {}'.format(best_val_score))

                    save_checkpoint(args, model, infos, optimizer, histories)
                    if args.save_history_ckpt:
                        save_checkpoint(args, model, infos, optimizer,
                            append=str(iteration))

                    if best_flag:
                        save_checkpoint(args, model, infos, optimizer, append='best')
                    
                    if args.save_every_log_step:
                        save_checkpoint(args, model, infos, optimizer, append=str(iteration))
                # break

                if iteration >= args.warmup_train_step:
                    break
        
        del model


    ###################
    ####   test   ####
    ###################
    if args.do_test:

        if args.load_best:
            model = configure_model_loading(args)
            logger.info('Load pretrained weights at {}'.format(os.path.join(args.save_checkpoint_path, 'model-best.pth')))
            fc_path = os.path.join(args.save_checkpoint_path, 'model-%s_fc_layers.pth' %('best'))
            fc_layers_state_dict = torch.load(fc_path)
            model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
            model.my_all_f1.load_state_dict(fc_layers_state_dict['my_all_f1'])
            model.my_f.load_state_dict(fc_layers_state_dict['my_f'])
        else:
            model = configure_model_loading(args)
            logger.info('Load pretrained weights at {}'.format(os.path.join(args.save_checkpoint_path, 'model-%s_fc_layers.pth' %(args.load_ckpt_num))))
            fc_path = os.path.join(args.save_checkpoint_path, 'model-%s_fc_layers.pth' %(args.load_ckpt_num))  # 这里要改
            fc_layers_state_dict = torch.load(fc_path)
            model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
            model.my_all_f1.load_state_dict(fc_layers_state_dict['my_all_f1'])
            model.my_f.load_state_dict(fc_layers_state_dict['my_f'])
    
        # model = configure_model_loading(args)
        # fc_path = os.path.join(args.save_checkpoint_path, 'model-%s_fc_layers.pth' %('best'))
        # fc_layers_state_dict = torch.load(fc_path)
        # model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
        # model.my_all_f1.load_state_dict(fc_layers_state_dict['my_all_f1'])
        # model.my_f.load_state_dict(fc_layers_state_dict['my_f'])

        # 是否将fc层转换为fp32
        if args.my_fc_fp32:
            model = convert_my_layers_to_fp32(model)

        logger.info('Load pretrained weights at {}'.format(os.path.join(args.start_from, 'model-best.pth')))
        
        # test_top_k_list = [10, 50, 100]
        # test_top_p_list = [0.95, 0.9, 0.85]

        test_top_k_list = [50]
        test_top_p_list = [0.9]

        for test_top_k, test_top_p in zip(test_top_k_list, test_top_p_list):
            
            logger.info('testing on topk {} top_p {}'.format(test_top_k, test_top_p))

            # test_generation_config = GenerationConfig(
            # min_new_tokens=args.min_new_tokens,
            # max_new_tokens=args.max_new_tokens, 
            # early_stopping=False,  # early stopping is only effective in beam search
            # do_sample=args.do_sample,
            # num_beams=args.num_beams,
            # top_k=args.test_top_k,
            # top_p=args.test_top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
            # temperature=args.temperature,
            # repetition_penalty=args.repetition_penalty,
            # context_aware_decoding_alpha=args.context_aware_decoding_alpha,
            # cad_full=args.cad_full,
            # cad_salience=args.cad_salience,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # pad_token_id=tokenizer.pad_token_id,
            # )
            test_generation_config = GenerationConfig(
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens, 
                early_stopping=False,  # early stopping is only effective in beam search
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_k=args.test_top_k,
                top_p=args.test_top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                context_aware_decoding_alpha=args.context_aware_decoding_alpha,
                cad_full=args.cad_full,
                cad_salience=args.cad_salience,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                ablation_main_sequence=args.ablation_main_sequence,
                ablation_presumm_sequence=args.ablation_presumm_sequence,
                ablation_null_sequence=args.ablation_null_sequence,
            )
            
            # Load pretrained weights
            if args.start_from is not None and os.path.isfile(os.path.join(args.start_from, 'model-best.pth')):
                model.load_state_dict(torch.load(os.path.join(args.start_from, 'model-best.pth')))
            results = test(args, all_test_set, logger, tokenizer, DEVICE, model, test_generation_config)
            # logger.info('rougeLsum_fmeasure: {}'.format(rougeL_score))

