import argparse
import os
import sys
import json
from datetime import datetime
import torch
import numpy as np
import random
from transformers import DistilBertTokenizer, DistilBertModel, MPNetTokenizer, GPT2Tokenizer
from tensorboardX import SummaryWriter

sys.path.append('../')
import utils.tensorboard_utils as TB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt', default='gpt2', type=str)
    parser.add_argument('--model', default='dec', type=str)
    parser.add_argument('--seed', default=888,type=int)
    parser.add_argument('--language_model', default='gpt', type=str)
    parser.add_argument('--dataset', default='mad', type=str)
    parser.add_argument('--video_filter', default=1, type=int)
    parser.add_argument('--num_frames', default=8, type=int)
    parser.add_argument('--num_latents', default=10, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-7, type=float)
    parser.add_argument('--loss', default='nce', type=str)
    parser.add_argument('--schedule', default=[10000], nargs='*', type=int, 
        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--clip_grad', default=0, type=float)
    parser.add_argument('--prefix', default='tmp', type=str)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--unit_test', default=None, type=str)
    parser.add_argument('-j', '--num_workers', default=2, type=int)
    parser.add_argument('--train_what', default='all', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('--sim', default='cos', type=str)
    parser.add_argument('--sentence_mode', default='cls', type=str)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--runtime_save_iter', default=1000, type=int)
    parser.add_argument('--aux_loss', default=1, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--clip_temp_mode', default='avg', type=str)
    parser.add_argument('--optim_policy', default='default', type=str)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--amp', default=1, type=int)
    parser.add_argument('--num_clips', default=4, type=int)
    parser.add_argument('--num_hypo', default=1, type=int)
    parser.add_argument('--use_context', default=1, type=int)
    parser.add_argument('--use_border_adapter', default=0, type=int)
    parser.add_argument('--remove_vision', default=0, type=int)
    parser.add_argument('--use_charbank', default=0)
    parser.add_argument('--use_unl', default=0, type=int)
    parser.add_argument('--lookahead', default=0, type=int)
    parser.add_argument('--rephrase', default=0, type=int)
    parser.add_argument('--use_bos', default=1, type=int)

    parser.add_argument('--num_history', default=64, type=int)
    parser.add_argument('--version', default='raw', type=str)
    parser.add_argument('--test_version', default=None, type=str)
    parser.add_argument('--clip_version', default='B32', type=str)
    parser.add_argument('--dev_split', default=0, type=int)

    parser.add_argument('--backprop_freq', default=1, type=int)
    parser.add_argument('--test_num_clips', default=1, type=int)
    parser.add_argument('--perceiver_type', default=2, type=int)
    parser.add_argument('--context_perceiver_type', default=0, type=int)
    parser.add_argument('--subtitle_perceiver_type', default=0, type=int)
    parser.add_argument('--context_feature_type', default='gpt', type=str)
    
    parser.add_argument('--freezeBN', action='store_true')
    parser.add_argument('--downstream', action='store_true')
    parser.add_argument('--single_video', action='store_true')
    parser.add_argument('--single_align_video', action='store_true')
    parser.add_argument('--cross_video', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--convert_from_frozen_bn', action='store_true')
    parser.add_argument('--keep_bn_eval', action='store_true')
    parser.add_argument('--init_s3d', action='store_true')
    parser.add_argument('--extract_feature', action='store_true')
    parser.add_argument('--feature_root', default='feature_coin/timesformer_8f_1fps', type=str)

    parser.add_argument('--test_mode', default='default-val', type=str)
    parser.add_argument('--save_video', action='store_true')

    args = parser.parse_args()
    return args


def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string

    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: 
        if os.path.dirname(args.test).endswith('model'):
            exp_path = os.path.dirname(os.path.dirname(args.test))
        else:
            exp_path = os.path.dirname(args.test)
    else:
        name_prefix = f"{args.name_prefix}_" if args.name_prefix else ""
        unit_test_tag = f'[unit-test-{args.unit_test}]' if args.unit_test else ''
        clip_tag = f'clips{args.num_clips}_' if args.dataset == 'mad' else ""
        context_tag = f'Context{args.num_history}_' if args.use_context else ''
        if args.context_perceiver_type != 0:
            context_tag += f'{args.context_feature_type}_'
        version_tag = f'_{args.version}' if args.dataset == 'mad' else ''
        CLIP_tag = f'ViT-{args.clip_version}'
        no_vision_tag = f'NoVision_' if args.remove_vision else ''
        rephrase_tag = f'Rephrase_' if args.rephrase else ''
        exp_path = (f"log-{args.prefix}/{name_prefix}{no_vision_tag}{context_tag}{unit_test_tag}{dt_string}_"
            f"{args.model}-{args.gpt}-P{args.perceiver_type}C{args.context_perceiver_type}S{args.subtitle_perceiver_type}_BOS{args.use_bos}_layer{args.num_layers}_latent{args.num_latents}_"
            f"Loss-{args.loss}_CharBank{args.use_charbank}_{rephrase_tag}Ahead{args.lookahead}_{args.language_model}_"
            f"token-{args.sentence_mode}_sim-{args.sim}_hypo{args.num_hypo}_{args.dataset}{version_tag}_{CLIP_tag}_DEV{args.dev_split}_{clip_tag}frames{args.num_frames}_"
            f"policy-{args.optim_policy}_"
            f"bs{args.batch_size}_lr{args.lr}")

    pre_prefix = ''
    log_path = os.path.join(pre_prefix, exp_path, 'log')
    model_path = os.path.join(pre_prefix, exp_path, 'model')
    exp_path = os.path.join(pre_prefix, exp_path)
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, model_path, exp_path


def setup(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if torch.cuda.is_available():
        if args.gpu is None:
            args.gpu = str(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        device = torch.device('cuda')

        num_gpu = len(str(args.gpu).split(','))
        args.num_gpu = num_gpu
        args.batch_size = num_gpu * args.batch_size
        print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.log_path, args.model_path, args.exp_path = set_path(args)

    writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'),
                                flush_secs=60)
    args.train_plotter = TB.PlotterThread(writer_train)
    writer_val = SummaryWriter(logdir=os.path.join(args.log_path, 'val'),
                            flush_secs=60)
    args.val_plotter = TB.PlotterThread(writer_val)

    # re-write language_model if use CLIP
    if args.model == 'clip':
        args.language_model = 'clip'
    elif args.model == 'timesformer':
        args.language_model = 'mpnet'

    if args.language_model == 'bert':
        args.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.language_model in ['distilbert']:
        args.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif args.language_model == 'mpnet':
        args.tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    elif args.language_model == 'clip':
        import clip
        class Tokenizer():
            def __call__(self, str_list, return_tensors='pt', **kwargs):
                token = clip.tokenize(str_list, truncate=True)
                if return_tensors != 'pt':
                    token = token.numpy()
                return {'input_ids': token}
        args.tokenizer = Tokenizer()
    elif args.language_model == 'gpt':
        args.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    args.iteration = 1
    args.stop_token_int = args.tokenizer(".").input_ids[0]

    if '/srun' in os.environ['_']:  # in sbatch
        print('running command: {')
        for key, item in args.__dict__.items():
            print(f'  "{key}": {item}')
        print('}')

    return device



def optim_policy(args, model, policy='default', version='gpt2'):
    params = []
    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale']
    param_group_no_decay = []
    param_group_with_decay = []

    if policy == 'default':
        ### only train xattn module, fix other gpt weights
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'Param not requires_grad: {name}')
                continue
            if ('gpt.' in name) and ('.xattn' not in name):
                continue  ## never touch gpt weights
            print(f'Param to optimize: {name}')
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
        params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})

    elif policy == 'pos':
        ### only train xattn module, fix other gpt weights, except pos embedding
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'Param not requires_grad: {name}')
                continue
            if ('gpt.' in name) and ('wpe.' not in name) and ('.xattn' not in name):
                continue  ## never touch gpt weights
            print(f'Param to optimize: {name}')
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
        params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    
    elif policy == 'all':  # train all gpt weights
        raise NotImplementedError
    
    elif policy == 'half':  # train half gpt blocks
        if version == 'gpt2':
            NUM_BLOCKS = 12
        elif version == 'gpt2-medium':
            NUM_BLOCKS = 24
        freeze_blocks = ['gpt.transformer.wte', 'gpt.transformer.wpe',] + \
            [f'gpt.transformer.h.{i}.' for i in range(int(NUM_BLOCKS//2))]

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'Param not requires_grad: {name}')
                continue
            if ('gpt.' in name) and ('.xattn' not in name):
                if any([i in name for i in freeze_blocks]):
                    continue
            print(f'Param to optimize: {name}')
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)
        params.append({'params': param_group_no_decay, 'lr': args.lr, 'weight_decay': 0.0})
        params.append({'params': param_group_with_decay, 'lr': args.lr, 'weight_decay': args.wd})
    
    else:
        raise NotImplementedError

    return params