import os
import sys
import torch
import torch.nn as nn
import math
import functools
import random 
import numpy as np
import torch.cuda.amp as amp 
import time
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from einops import rearrange, repeat 
import math
from tqdm import tqdm
import string
import json
import pickle
import pandas as pd

from config import parse_args, set_path, setup, optim_policy
sys.path.append('../gpt/')
from gpt_model import TemporalDecoder
from name_loader import LSMDC_NameLoader, MAD_NameLoader, MovieNet_NameLoader

sys.path.append('../')
import utils.tensorboard_utils as TB
from utils.train_utils import clip_gradients, in_sbatch, set_worker_sharing_strategy
from utils.data_utils import DataLoaderBG
from utils.utils import AverageMeter, AverageMeter_Raw, save_checkpoint, \
calc_topk_accuracy, ProgressMeter, neq_load_customized, save_runtime_checkpoint



class CharRecog(nn.Module):
    def __init__(self, dim, num_layers, use_proj=False):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        assert dim % 64 == 0
        self.tfm = TemporalDecoder(width=dim, layers=num_layers, heads=dim//64)
        self.pos_enc = nn.Embedding(256, embedding_dim=dim)
        nn.init.normal_(self.pos_enc.weight, mean=0, std=0.5)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)
        nn.init.normal_(self.classifier.weight, std=0.01)
        self.use_proj = use_proj
        if use_proj:
            self.face_proj = nn.Linear(dim, dim)
            nn.init.normal_(self.face_proj.weight, std=0.01)
            nn.init.zeros_(self.face_proj.bias)

    def forward(self, face, face_padding_mask, visual):
        """face: B,10,C
        face_padding_mask: B,10
        visual: B,T,C"""
        # warning: tfm expects SEQ, B, C
        T = visual.shape[1]
        pos = self.pos_enc.weight[None, 0:T]
        if self.use_proj:
            face = self.face_proj(face)
            visual = self.face_proj(visual)
        out = self.tfm(x=face.transpose(0,1), memory=visual.transpose(0,1), tgt_key_padding_mask=face_padding_mask, pos=pos.transpose(0,1))
        out = out[-1].transpose(0,1)
        logits = self.classifier(self.norm(out))
        return logits


def train(loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args, val_loader=None):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    progress = ProgressMeter(
        len(loader), [batch_time, data_time],
        prefix='Epoch:[{}]'.format(epoch))
    model.train()
    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    for idx, input_data in enumerate(loader):
        data_time.update(time.time() - end)
        video_seq = input_data['video'].to(device, non_blocking=True)
        B,N,T,C = video_seq.shape
        video_seq = rearrange(video_seq, 'b n t c -> (b n) t c')
        loss_dict = {}

        exemplar_feature = input_data['exampler_feature'].to(device, non_blocking=True)
        exemplar_feature = rearrange(exemplar_feature, 'b n t c -> (b n) t c')
        exemplar_attn_mask = input_data['exampler_attn_mask'].to(device, non_blocking=True)
        exemplar_attn_mask = rearrange(exemplar_attn_mask, 'b n t -> (b n) t')

        # char_text = input_data['char_text']
        tgt_text = input_data['text']
        # tgt = text_to_label(char_text, tgt_text)
        tgt = input_data['binary_tgt'].to(device, non_blocking=True)
        tgt = rearrange(tgt, 'b n t -> (b n) t')

        logits = model(exemplar_feature, face_padding_mask= ~exemplar_attn_mask.bool(), visual=video_seq)
        assert logits.shape[2] == 1
        logits_flatten = logits[:,:,0][exemplar_attn_mask.bool()]
        tgt_flatten = tgt[exemplar_attn_mask.bool()]

        # # over sampling
        # N_tgt = tgt_flatten.float().sum().item()
        # random_mask = torch.rand_like(tgt_flatten.float())
        # random_mask = random_mask * (1-tgt_flatten)
        # _, chosen_idx = torch.topk(random_mask, k=int(N_tgt))
        # chosen_mask = tgt_flatten.clone()
        # chosen_mask.scatter_(0, chosen_idx, 1)
        # logits_flatten = logits_flatten[chosen_mask.bool()]
        # tgt_flatten = tgt_flatten[chosen_mask.bool()]

        if tgt_flatten.float().mean().item() > 0:
            weight_flatten = torch.ones_like(logits_flatten) * 0.5/(1-tgt_flatten.float().mean())
            weight_flatten.masked_fill_(tgt_flatten.bool(), value=0.5/tgt_flatten.float().mean())
        else:
            print('warning: all zero labels')
            weight_flatten = torch.ones_like(logits_flatten)
        loss = F.binary_cross_entropy_with_logits(logits_flatten, tgt_flatten.float(), weight=weight_flatten)

        prec = ((logits_flatten > 0) * tgt_flatten).float().sum() / torch.clamp((logits_flatten > 0).float().sum(), min=1e-5)
        recall = ((logits_flatten > 0) * tgt_flatten).float().sum() / torch.clamp(tgt_flatten.float().sum(), min=1e-5)
        loss_dict = {'loss': loss.detach(), 'prec': prec, 'recall': recall}

        if idx == 0:
            avg_meters = {k: AverageMeter(f'{k}:',':.4f') for k in loss_dict.keys()}
        for metric, value in loss_dict.items():
            avg_meters[metric].update(value.item(), B)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        progress.display(idx)
        print('\t' + ' '.join([f"{k}:{v.item():.3f}" for k,v in loss_dict.items()]))
        lr_scheduler.step()

        if args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'local/{k}', v.item(), args.iteration)

        end = time.time()
        args.iteration += 1
    
    print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')
    for metric_name, avg_meter in avg_meters.items():
        args.train_plotter.add_data(f'global/{metric_name}', avg_meter.avg, epoch)
    return losses.avg


@torch.no_grad()
def evaluate(loader, model, device, epoch, args, prefix=None):
    model.eval()
    if args.test:
        all_predictions = []

    for idx, input_data in tqdm(enumerate(loader), total=len(loader)):
        video_seq = input_data['video'].to(device, non_blocking=True)
        B,N,T,C = video_seq.shape
        video_seq = rearrange(video_seq, 'b n t c -> (b n) t c')

        exemplar_feature = input_data['exampler_feature'].to(device, non_blocking=True)
        exemplar_feature = rearrange(exemplar_feature, 'b n t c -> (b n) t c')
        exemplar_attn_mask = input_data['exampler_attn_mask'].to(device, non_blocking=True)
        exemplar_attn_mask = rearrange(exemplar_attn_mask, 'b n t -> (b n) t')

        if isinstance(loader.dataset, LSMDC_NameLoader):
            char_text = input_data['char_text']
            tgt_text = input_data['text']
            tgt = text_to_label(char_text, tgt_text)
            tgt = tgt.to(device, non_blocking=True)
            tgt = rearrange(tgt, 'b n t -> (b n) t')
        else:
            # char_text = input_data['char_text']
            tgt_text = input_data['text']
            # tgt = text_to_label(char_text, tgt_text)
            tgt = input_data['binary_tgt'].to(device, non_blocking=True)
            tgt = rearrange(tgt, 'b n t -> (b n) t')

        logits = model(exemplar_feature, face_padding_mask= ~exemplar_attn_mask.bool(), visual=video_seq)
        assert logits.shape[2] == 1
        logits_flatten = logits[:,:,0][exemplar_attn_mask.bool()]
        tgt_flatten = tgt[exemplar_attn_mask.bool()]

        if tgt_flatten.float().mean().item() > 0:
            weight_flatten = torch.ones_like(logits_flatten) * 0.5/(1-tgt_flatten.float().mean())
            weight_flatten.masked_fill_(tgt_flatten.bool(), value=0.5/tgt_flatten.float().mean())
        else:
            print('warning: all zero labels')
            weight_flatten = torch.ones_like(logits_flatten)
        loss = F.binary_cross_entropy_with_logits(logits_flatten, tgt_flatten.float(), weight=weight_flatten)

        prec = ((logits_flatten > 0) * tgt_flatten).float().sum() / torch.clamp((logits_flatten > 0).float().sum(), min=1e-5)
        recall = ((logits_flatten > 0) * tgt_flatten).float().sum() / torch.clamp(tgt_flatten.float().sum(), min=1e-5)
        loss_dict = {'loss': loss.detach(), 'prec': prec, 'recall': recall}

        if idx == 0:
            avg_meters = {k: AverageMeter(f'{k}:',':.4f') for k in loss_dict.keys()}
        for metric, value in loss_dict.items():
            avg_meters[metric].update(value.item(), B)

        if args.test:
            probs = logits[:,:,0].sigmoid()
            probs = probs.masked_fill(~exemplar_attn_mask.bool(), -1)
            for n_idx in range(N):
                if isinstance(loader.dataset, LSMDC_NameLoader):
                    start_log = input_data['start'][0][n_idx]
                    end_log = input_data['end'][0][n_idx]
                else:
                    start_log = input_data['start'][0][n_idx]
                    end_log = input_data['end'][0][n_idx]
                all_predictions.append({'vid': input_data['vid'][0], 
                                        'prob': probs[n_idx].tolist(),
                                        'start': start_log, 
                                        'end': end_log,
                                        'movienet_tgt': tgt_flatten.tolist()})

    print(' '.join([f'{metric_name}: {avg_meter.avg}' for metric_name, avg_meter in avg_meters.items()]))
    if args.test:
        for pred_item in all_predictions:
            if isinstance(pred_item['vid'], np.int64):
                pred_item['vid'] = int(pred_item['vid'])
        import ipdb; ipdb.set_trace()
        with open('MAD_eval_prob.json', 'w') as fobj:
            json.dump(all_predictions, fobj)
        sys.exit(0)

        # with open('MAD_train_prob.json', 'w') as fobj: json.dump(all_predictions, fobj)

        # char_prob_dict = dict(tuple(pd.DataFrame.from_records(all_predictions).groupby('vid')))
        # with open('MAD_char_prob_dict_train.pkl', 'wb') as fobj:
        #     pickle.dump(char_prob_dict, fobj)

    for metric_name, avg_meter in avg_meters.items():
        args.val_plotter.add_data(f'global/{metric_name}', avg_meter.avg, epoch)
    return avg_meters['loss'].avg


def text_to_label(char_text, tgt_text):
    assert len(char_text) == len(tgt_text)
    B = len(char_text)
    N = len(char_text[0])
    tgt_tensor = torch.zeros(B, N, 10, dtype=torch.long)
    for b_idx, (char_t, tgt_t) in enumerate(zip(char_text, tgt_text)):
        assert len(char_t) == len(tgt_t)
        for t_idx, (char, tgt) in enumerate(zip(char_t, tgt_t)):
            if 'unknown' in tgt:
                continue
            tgt_list = tgt.split(',')
            tgt_list = [rm_punct(t).strip() for t in tgt_list]
            tgt_list = [t for t in tgt_list if len(t) > 0]
            if len(tgt_list) == 0:
                continue
            tgt_array = np.array(tgt_list)
            
            char_list = char.split('possible characters:')[-1]
            char_list = char_list.split('<image>')
            char_list = [rm_punct(c).strip() for c in char_list]
            char_list = [c for c in char_list if len(c) > 0]
            char_array = np.array(char_list)
            num_C = char_array.shape[0]

            tgt_tensor[b_idx, t_idx, 0:num_C] = torch.tensor(((char_array[:, None] == tgt_array[None, :]).astype(int).sum(-1) > 0).astype(int))    
    return tgt_tensor


translator_rm_punct = str.maketrans('', '', string.punctuation)
def rm_punct(s):
    new_string = s.translate(translator_rm_punct)
    return new_string


def get_dataset(args):
    batch_size = -1
    train_mode = 'train'
    val_mode = 'val'
    tokenizer = args.tokenizer

    if args.dataset == 'lsmdc_name':
        trainD = LSMDC_NameLoader
    elif args.dataset == 'mad_name':
        trainD = MAD_NameLoader
    elif args.dataset == 'movienet_name':
        trainD = MovieNet_NameLoader
    train_dataset = trainD(
        tokenizer=tokenizer,
        mode=train_mode,
        num_frames=args.num_frames,
        num_clips=args.num_clips,
        batch_size=batch_size,
        version=args.version,
        return_gpt_feature=False,  # args.context_perceiver_type!=0,
        clip_version=args.clip_version,
        return_subtitle_gpt_feature=args.subtitle_perceiver_type!=0,
        context_feature_type=args.context_feature_type,
        use_charbank=args.use_charbank,
        lookahead=args.lookahead,
        rephrase=args.rephrase,
        load_history=int(args.perceiver_type==4),
        force_resample=True,
        )
    
    valD = MovieNet_NameLoader
    val_dataset = valD(
        tokenizer=tokenizer,
        mode=val_mode,
        num_frames=args.num_frames,
        num_clips=16,  # args.num_clips,
        batch_size=batch_size,
        version='lsmdc_named',  # args.version,
        return_gpt_feature=False,  # args.context_perceiver_type!=0,
        clip_version=args.clip_version,
        return_subtitle_gpt_feature=args.subtitle_perceiver_type!=0,
        context_feature_type=args.context_feature_type,
        use_charbank=args.use_charbank,
        lookahead=args.lookahead,
        load_history=int(args.perceiver_type==4),
        force_resample=True,
        )

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoaderBG(train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler, 
    )

    val_sampler = torch.utils.data.SequentialSampler(val_dataset) 
    val_bs = args.batch_size
    val_loader = DataLoaderBG(val_dataset,
        batch_size=val_bs, num_workers=args.num_workers//2,
        collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=False,
        shuffle=(val_sampler is None), sampler=val_sampler, 
    )
    return train_dataset, val_dataset, train_loader, val_loader


def get_model_card(tag):
    model_card = {}
    return model_card.get(tag, tag), tag


def main(args):
    device = setup(args)
    if args.clip_version == 'L14':
        visual_dim = 768
    else:
        visual_dim = 512
    model = CharRecog(dim=visual_dim, 
                      num_layers=args.num_layers,
                      use_proj=True)
    model.to(device)
    model_without_dp = model

    ### test ###
    if args.test:
        print(f"test from checkpoint {args.test}")
        args.test, _ = get_model_card(args.test)
        if os.path.exists(args.test):
            checkpoint = torch.load(args.test, map_location='cpu')
            state_dict = checkpoint['state_dict']
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint.get('best_acc', 0)
            try:
                model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] Non-Equal load for resuming training!')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
        else:
            print(f'{args.test} does not exists, test random init?')
            import ipdb; ipdb.set_trace()
            args.start_epoch = 1
            args.iteration = 0

        unit_test_feature_root = None
        print(f'test with {args.test_mode} mode')

        # D = LSMDC_NameLoader
        # D = MovieNet_NameLoader
        # test_mode = 'test'

        # for MAD-TRAIN movies:
        # D = MAD_NameLoader; test_mode = 'train'

        # for MAD-EVAL movies:
        D = LSMDC_NameLoader; test_mode = 'test'

        test_dataset = D(
            tokenizer=args.tokenizer,
            mode=test_mode,
            num_frames=args.num_frames,
            num_clips=args.num_clips,
            unit_test_feature_root=unit_test_feature_root,
            version=args.version,
            clip_version=args.clip_version,
            test_version=args.test_version,
            return_gpt_feature=False,
            return_subtitle_gpt_feature=args.subtitle_perceiver_type!=0,
            context_feature_type=args.context_feature_type,
            use_charbank=args.use_charbank,
            lookahead=args.lookahead,
            load_history=int(args.perceiver_type==4),
            force_resample=True
            )
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

        loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=1, num_workers=args.num_workers,
            collate_fn=test_dataset.collate_fn, pin_memory=True, drop_last=False,
            shuffle=False, sampler=test_sampler,
            worker_init_fn=set_worker_sharing_strategy
        )
        evaluate(loader, model, device, args.start_epoch, args)
        sys.exit(0)

    ### dataset ###
    _, _, train_loader, val_loader = get_dataset(args)

    ### optimizer ###
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    ### restart ###
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume, _ = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        if args.convert_from_frozen_bn:
            tmp_state_dict = {}
            for k,v in state_dict.items():
                if '.bn' in k:
                    tmp_state_dict[k.replace('.scale', '.weight')] = v
                else:
                    tmp_state_dict[k] = v
            state_dict = tmp_state_dict

        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training, continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f'Not resuming optimizer states due to Error: {e}\nInitialized the optimizer instead...')
    ### restart ###

    args.decay_steps = args.epochs * len(train_loader)
    args.warmup_epochs = float(args.epochs / 20)
    def lr_schedule_fn(iteration, iter_per_epoch, args):
        if iteration < args.warmup_epochs * iter_per_epoch:
            lr_multiplier = iteration / (args.warmup_epochs * iter_per_epoch)
        else:
            lr_multiplier = 0.5 * \
                (1. + math.cos(math.pi * (iteration - args.warmup_epochs*iter_per_epoch) / (args.epochs*iter_per_epoch - args.warmup_epochs*iter_per_epoch)))
        return lr_multiplier
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_schedule_fn, iter_per_epoch=len(train_loader), args=args)
    )
    if args.resume:
        lr_scheduler.step(args.iteration)  # for resume mode
    grad_scaler = amp.GradScaler()
    torch.manual_seed(0)

    best_acc = 100
    evaluate(val_loader, model, device, -1, args)
    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        train(train_loader, model, optimizer, lr_scheduler, grad_scaler, device, epoch, args)
        val_loss = evaluate(val_loader, model, device, epoch, args)
        if (epoch % args.eval_freq == 0) or (epoch == args.epochs - 1):
            # is_best = val_loss < best_acc  # temporary use val loss
            is_best = False  # rewritten
            best_acc = min(val_loss, best_acc)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, args.eval_freq, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=True)
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)



"""python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5

python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5 --clip_version L14

python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5 --clip_version L14 --dataset mad_name

python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5 --clip_version L14 --dataset movienet_name


# inference:
python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5 --clip_version L14 \
    --test ~/beegfs/DATA/AutoAD/char_recog/log-tmp/2023_03_05_16_34_dec-gpt2-P2C0S0_BOS1_layer2_latent10_Loss-nce_CharBankglobal-ce_Ahead2_gpt_token-cls_sim-cos_hypo1_mad_raw_ViT-L14_DEV0_clips8_frames8_policy-default_bs64_lr0.0003/model/epoch9.pth.tar  --epochs 10  --num_clips 32

python recog_main.py --batch_size 64 --num_clips 8 --lookahead 2 --use_context 0 --use_charbank global-ce -j 8 --epochs 5 --clip_version L14 \
    --test log-tmp/Correct_2023_07_20_12_25_dec-gpt2-P2C0S0_BOS1_layer2_latent10_Loss-nce_CharBankglobal-ce_Ahead2_gpt_token-cls_sim-cos_hypo1_movienet_name_ViT-L14_DEV0_frames8_policy-default_bs64_lr0.0001/model/epoch4.pth.tar    

"""