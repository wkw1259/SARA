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
from transformers import get_linear_schedule_with_warmup


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
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    # 只存fc层
    fc_layers_state_dict = {
    'my_all_f': model.my_all_f.state_dict(),
    'my_alpha_f': model.my_alpha_f.state_dict(),
    'my_beta_f': model.my_beta_f.state_dict(),
    'my_gamma_f': model.my_beta_f.state_dict(),
    }

    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s_fc_layers.pth' %(append))
    torch.save(fc_layers_state_dict, checkpoint_path)
    logger.info("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
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

def get_self_critical_reward(greedy_res, data_gts, gen_result, tokenizer, RougeL_reward_weight, Rouge1_reward_weight, Rouge2_reward_weight, factkb_weight):
    
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()
    # data_gts = data_gts.data.cpu().numpy()
    
    # 把训练生成的greedy的 和 测试的拼到一起
    for i in range(gen_result_size):
        res[i] = [gen_result[i]]
    for i in range(batch_size):
        res[gen_result_size + i] = [greedy_res[i]]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        # gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
        gts[i] = [data_gts[i]]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    
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
    scores, Rouge1_scores, Rouge2_scores = [], [], []
    evaluator = Evaluator()
    
    # 逐条计算rougeLsum_fmeasure分数
    for pred, gt in zip(res_for_evaluate, gt_for_evaluate):
        result_dict = evaluator.evaluate(pred, gt, None, metrics=["rouge"])
        scores.append(result_dict['rougeLsum_fmeasure'])
        Rouge1_scores.append(result_dict['rouge1_fmeasure'])
        Rouge2_scores.append(result_dict['rouge2_fmeasure'])

    # for k, v in result_dict.items():
    #     print(f"{k} -> {v*100:.1f}")
    # for k, v in rouge_dict.items():
    #     result_dict[k] = v
    
    scores, Rouge1_scores, Rouge2_scores = np.array(scores), np.array(Rouge1_scores), np.array(Rouge2_scores)
    scores = RougeL_reward_weight * scores + Rouge1_reward_weight * Rouge1_scores + Rouge2_reward_weight * Rouge2_scores
    
    # scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

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
    dataset = test_set
    num_batch = len(dataset)//args.batch_size+1 if len(dataset)%args.batch_size!=0 else len(dataset)//args.batch_size

    # if args.context_aware_decoding_alpha > 0.:
    #     null_input = get_null_input_decoder(args.dataset)

    logger.info("start decoding!")
    predictions, references, documents = [], [], []
    start_time = time.time()
    total_decoding_length = 0
    for batch_idx, _ in tqdm(enumerate(range(num_batch))):
        batch = dataset[args.batch_size*batch_idx: args.batch_size*(batch_idx+1)]

        batch_input, batch_reference, batch_presumm = [row[0] for row in batch], [row[1] for row in batch], [row[2] for row in batch]
        tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=2048, padding=True, truncation=True)
        if args.context_aware_decoding_alpha >=0.: #full and salience and prompt
            print('full+salience-prompt generation')
            batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
            tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", padding=True) 
            batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
            tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", padding=True)
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                    null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                )
                
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
            tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", padding=True)
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
        # break
    
    print(f"total decoding takes {time.time()-start_time:.1f} seconds!")
    print(f"average token per seconds -> {(time.time()-start_time)/total_decoding_length:.5f}")

    # del model  # offload model to avoid memory spike

    assert len(predictions)==len(references), "mismatched shape, force exit!"
    evaluator = Evaluator()
    result_dict = evaluator.evaluate(predictions, references, documents, metrics=["rouge", "bertscore", "factkb"])
    for k, v in result_dict.items():
        logger.info(f"{k} -> {v*100:.1f}")
    logger.info("\n")

    # if args.save_output:
    #     if not args.output_fname:
    #         model_name = config._name_or_path.split("/")[-1]
    #         args.output_fname = f"{model_name}_max_new_tokens_{args.max_new_tokens}_topk_{args.top_k}_topp_{args.top_p}_alpha_{args.context_aware_decoding_alpha}.jsonl"
    #         args.output_fname = os.path.join("./generation", args.output_fname)

    #     with open(args.output_fname, "w") as fout:
    #         for i in range(len(predictions)):
    #             json_line = {"prediction": predictions[i], "reference:": references[i]}
    #             json.dump(json_line, fout)
    #             fout.write("\n")
    #     fout.close()
    #     logger.info(f"generation file -> {args.output_fname}\n")
    
    return result_dict['rougeLsum_fmeasure']  # 返回rouge-L
    
    

if __name__ == "__main__":
    
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--loading_mode", type=str, default="fp32")
    parser.add_argument("--min_new_tokens", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    parser.add_argument("--cad_full", type=float, default=1.0)
    parser.add_argument("--cad_salience", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_fname", type=str, default="")
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')  
    parser.add_argument('--save_checkpoint_every', type=int, default=25,  # 2500
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='model_output',
                    help='directory to store checkpointed models')
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
    parser.add_argument('--alpha_et_hidden_size', type=int, default=200)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--RougeL_reward_weight', type=float, default=1.0)
    parser.add_argument('--Rouge1_reward_weight', type=float, default=0.0)
    parser.add_argument('--Rouge2_reward_weight', type=float, default=0.0)
    parser.add_argument('--factkb_weight', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument("--logging", type=str, default="./default_decoder.log")
    args = parser.parse_args()

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
    console_handler.setLevel(logging.INFO)
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

    train_set, validation_set, test_set = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    # train_set = pretokenize(train_set[:args.num_samples], tokenizer, args.max_input_length)
    train_set = pretokenize(train_set, tokenizer, args.max_input_length)
    train_set = [[template_input_decoder(row, args.dataset), row[1], row[2]] for row in train_set]
    
    # test_set = pretokenize(test_set[:2], tokenizer, args.max_input_length)
    test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)
    test_set = [[template_input_decoder(row, args.dataset), row[1], row[2]] for row in test_set]
    
    model = configure_model_loading(args)
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
    
    generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        cad_full=args.cad_full,
        cad_salience=args.cad_salience,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    ###################
    ####   train   ####
    ###################
    
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps//10),  # warmup步数取得总步数的10%
        num_training_steps=num_training_steps
    )

    logger.info("start training!")
    
    for epoch_i in range(args.epoch_num):
        # predictions, references, documents = [], [], []
        start = time.time()
        total_decoding_length = 0
        for batch_idx, _ in tqdm(enumerate(range(num_batch))):
            batch = dataset[args.batch_size*batch_idx: args.batch_size*(batch_idx+1)]

            batch_input, batch_reference, batch_presumm = [row[0] for row in batch], [row[1] for row in batch], [row[2] for row in batch]
            tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=2048, padding=True, truncation=True)
            if args.context_aware_decoding_alpha >=0.: #full and salience and prompt
                # print('full+salience-prompt generation')
                batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
                tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", padding=True) 
                batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
                tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", padding=True)
                
                tokenized_gt = tokenizer(batch_reference, return_tensors="pt", max_length=2048, padding=True, truncation=True)
                    
                model.eval()
                # with torch.no_grad():
                output_dict = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                    null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )   # output [bs 生成的seq]   scores 生成的seq * [16, 50257]  
                output_eval = output_dict.sequences
                greedy_res = output_eval[:, tokenized_input.input_ids.shape[1]:]  # 用生成之后的seq来做损失
                    
                # 要通过greedy生成多个
                model.train()
                optimizer.zero_grad()
                output_dict = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    presumm_input=tokenized_presumm.input_ids.to(DEVICE),
                    null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    presumm_attention_mask=tokenized_presumm.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_return_sequences=args.num_return_sequences
                )
                
                output = output_dict.sequences
                gen_result, scores = output[:, tokenized_input.input_ids.shape[1]:], output_dict.scores
                scores = torch.stack(scores, dim=0).permute(1, 0, 2)  #[bs*num_return_sequences 生成的seq 50257]
                ## TODO 是否需要做一个logsoftmax？？
                sample_logprobs = F.log_softmax(scores, dim=-1)
                
                reward = get_self_critical_reward(greedy_res, tokenized_gt.input_ids.to(DEVICE), gen_result, tokenizer, 
                                                  args.RougeL_reward_weight, args.Rouge1_reward_weight, args.Rouge2_reward_weight, args.factkb_weight)
                reward = torch.from_numpy(reward).to(sample_logprobs)   
                rl_crit = RewardCriterion()
                loss = rl_crit(sample_logprobs, gen_result.data, reward, reduction='mean')

                torch.cuda.synchronize()
                end = time.time()
                
                logger.info("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                        .format(iteration, epoch_i, loss.item(), end - start))

                train_loss = loss.item()
                if math.isinf(train_loss):
                    logger.info("loss == inf, continue")
                    continue

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # if opt.grad_clip_value != 0:
                #     getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
                optimizer.step()
                scheduler.step()

                # Update the iteration
                iteration += 1

                # Write the training loss summary
                if (iteration % args.losses_log_every == 0):
                    logger.info('loss_history: iteration: {} reward: {}'.format(iteration, loss))

                # update infos
                infos['iter'] = iteration
                          
            elif args.cad_salience >= 0.: #full and salience
                print('full+salience generation')
                batch_presumm = [presumm_input_decoder(row,args.dataset) for row in batch]
                tokenized_presumm = tokenizer(batch_presumm, return_tensors="pt", padding=True)
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
                rougeL_score = test(args, test_set, logger, tokenizer, DEVICE, model, generation_config)
                
                # 重置存储的结果
                # predictions, references, documents = [], [], []
                
                # Write validation result into summary
                # histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
                logger.info('evaluate rougeL: {}'.format(rougeL_score))
                
                # Save model if is improving on validation result
                current_score = rougeL_score

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
            # break
    
    del model
    ###################
    ####   test   ####
    ###################
    model = configure_model_loading(args)
    fc_path = os.path.join(args.checkpoint_path, 'model%s_fc_layers.pth' %('best'))
    fc_layers_state_dict = torch.load(fc_path)
    model.my_all_f.load_state_dict(fc_layers_state_dict['my_all_f'])
    model.my_alpha_f.load_state_dict(fc_layers_state_dict['my_alpha_f'])
    model.my_beta_f.load_state_dict(fc_layers_state_dict['my_beta_f'])
    model.my_gamma_f.load_state_dict(fc_layers_state_dict['my_gamma_f'])

    logger.info('Load pretrained weights at {}'.format(os.path.join(args.start_from, 'model-best.pth')))
    # Load pretrained weights
    if args.start_from is not None and os.path.isfile(os.path.join(args.start_from, 'model-best.pth')):
        model.load_state_dict(torch.load(os.path.join(args.start_from, 'model-best.pth')))
    rougeL_score = test(args, test_set, logger, tokenizer, DEVICE, model, generation_config)
    logger.info('rougeLsum_fmeasure', rougeL_score)