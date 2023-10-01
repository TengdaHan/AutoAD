import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import time
import re
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import json 
import lmdb
import math
import pickle
from glob import glob 
from joblib import delayed, Parallel
import h5py
import sys
import hashlib
import string
import ast


lsmdc_split_info = json.load(open(os.path.join('/work/htd/Desktop_tmp/AD-I/datasets', 'lsmdc_split.json')))
LSMDC_TRAIN = lsmdc_split_info['TRAIN']
LSMDC_VAL = lsmdc_split_info['VAL']
LSMDC_TEST = lsmdc_split_info['TEST']
LSMDC_TRAINVAL = lsmdc_split_info['TRAIN'] + lsmdc_split_info['VAL']

mad_split_info = json.load(open(os.path.join('/work/htd/Desktop_tmp/AD-I/datasets', 'mad_split.json')))
MAD_EVAL = mad_split_info['EVAL']
MAD_TRAIN = mad_split_info['TRAIN']

EOS_TOKEN_ID = 50256



def check_condition(items, condition_fn, desc=''):
    result = Parallel(n_jobs=8, prefer='threads')(delayed(condition_fn)(i) for i in tqdm(items, total=len(items), 
        desc="Check Condition" if desc == '' else desc, 
        leave=False))
    passed_items = []
    for res, item in zip(result, items):
        if res:
            passed_items.append(item)
    return passed_items


def check_existence(video_list, video_root, vid_to_path_dict, tmpdir='tmp'):
    """check existence of a list of files, support cache"""
    os.makedirs(os.path.join(os.path.dirname(__file__), tmpdir), exist_ok=True)
    hash_tag = hashlib.sha256((video_root+json.dumps(video_list)).encode()).hexdigest()
    hash_file = os.path.join(os.path.dirname(__file__), tmpdir, f'{hash_tag}.check_existence.json')
    if os.path.exists(hash_file):
        print(f'load from TMP file: {hash_file}')
        existed_video = json.load(open(hash_file))
    else:
        check_fn = lambda x: os.path.exists(os.path.join(video_root, vid_to_path_dict.get(x,x)))
        result = Parallel(n_jobs=8, prefer='threads')(delayed(check_fn)(i) for i in tqdm(
            video_list, total=len(video_list), desc="Check Existence", leave=False,
            disable=('/srun' in os.environ['_'])))
        existed_video = []
        for res, vid in zip(result, video_list):
            if res:
                existed_video.append(vid)
        
        with open(hash_file, 'w') as fobj:
            json.dump(existed_video, fobj)
    return existed_video


def pad_sequence_by_last(sequences):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, 0.0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor 
        out_tensor[i, length:, ...] = tensor[-1, ...]
    return out_tensor


def pad_sequence_to_length(sequences, length=32, value=0, pad_left=False):
    N = len(sequences)
    out_tensor = torch.empty(N, length, *sequences[0].shape[1::], dtype=sequences[0].dtype).fill_(value)
    for i, tensor in enumerate(sequences):
        len_ = tensor.size(0)
        if pad_left:
            if len_ <= length:
                out_tensor[i, -len_::] = tensor
            else:
                out_tensor[i, :] = tensor[-length::] 
        else:
            if len_ <= length:
                out_tensor[i, 0:len_] = tensor
            else:
                out_tensor[i, :] = tensor[0:length]
    return out_tensor


def pad_sequence_by_dim(sequences, dim, value=0):
    assert dim == 1  # only works for dim=1
    N = len(sequences)
    MAX_LEN = max([i.shape[dim] for i in sequences])
    out_shape = [N, *sequences[0].shape]
    out_shape[dim+1] = MAX_LEN
    out_tensor = torch.empty(*out_shape, dtype=sequences[0].dtype).fill_(value)
    for i, tensor in enumerate(sequences):
        len_ = tensor.shape[dim]
        out_tensor[i,:,0:len_] = tensor
    return out_tensor


translator_rm_punct = str.maketrans('', '', string.punctuation)
def rm_punct(s):
    new_string = s.translate(translator_rm_punct)
    return new_string


class LSMDC_NameLoader():
    def __init__(self,
                 mode='train', tokenizer=None, 
                 num_frames=8, num_clips=4,
                 use_charbank=0,
                 lookahead=0,
                 force_resample=False,
                 clip_version='B32',
                 exclude_movienet=False,
                 **kwargs):
        if len(kwargs):
            print(f'LSMDC_NameLoader: {kwargs} not used by dataset')

        version_card = {'lsmdc': '/scratch/shared/beegfs/htd/DATA/MAD/MAD_anno_dict_pd.pkl',  # someone
                        'lsmdc_named': '/scratch/shared/beegfs/htd/DATA/MAD/MAD_named_anno_dict_pd.pkl',  # named
                        }
        self.anno_named = pickle.load(open(version_card['lsmdc_named'], 'rb'))
        self.anno_someone = pickle.load(open(version_card['lsmdc'], 'rb'))
        # del self.anno_named['3021'], self.anno_someone['3021'] 
        self.movie_id_to_anno = {key.split('_')[0]: key for key in self.anno_named.keys()}
        self.lookahead = lookahead
        self.tokenizer = tokenizer
        self.answer_prompt = ' Characters in <video>:'
        self.exclude_movienet = exclude_movienet

        mode_conversion = {'test': 'val'}
        mode_ = mode_conversion.get(mode, mode)
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.mode = mode_
        self.real_mode = mode
        if mode_ == 'train':
            self.movie_names = LSMDC_TRAIN
            if exclude_movienet:
                pre_length = len(self.movie_names)
                movienet_anno = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/movienet/movienet_face_anno.csv')
                mvn_id = set(movienet_anno['movie_id'].unique().tolist())
                mad_imdb_info = pd.read_csv("/scratch/shared/beegfs/htd/DATA/MAD/mad_imdb_info.csv")
                mad_name_to_imdb = dict(zip(mad_imdb_info['movie_name'], mad_imdb_info['imdb']))
                tmp_movie_names = []
                for name in self.movie_names:
                    if mad_name_to_imdb.get(name) in mvn_id:
                        continue
                    tmp_movie_names.append(name)
                self.movie_names = tmp_movie_names
                print(f"Excluding MovieNet movies reduces movies from {pre_length} to {len(self.movie_names)}")
        else:
            # self.movie_names = LSMDC_VAL
            self.movie_names = MAD_EVAL

        # CharBank
        self.use_charbank = use_charbank
        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']: # give global charbank
            # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal'
            # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_2023mar'
            self.charbank_root = '/scratch/shared/beegfs/htd/MAD/charbank_cos_top10_cal_2023jul'


        self.clip_version = clip_version
        if clip_version == 'B32':
            self.clip_dim = 512
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_frames_features_5fps'
        elif clip_version == 'L14':
            self.clip_dim = 768
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_L14_frames_features_5fps'

        self.pre_sample_clips(force_resample=(force_resample or ('val' in mode_) or (self.real_mode=='test')))  # 'val' in mode_
        self.dataset_length = len(self.all_clips)
        print(f'Loaded {mode}-set from {self.frame_feature_root}')


    def pre_sample_clips(self, force_resample=False):
        # continuous clips from movies
        identify_string = f'LSMDC_Name_{self.mode}_{self.num_clips}_{self.use_charbank}'
        hash_tag = hashlib.sha256((identify_string+json.dumps(self.movie_names)).encode()).hexdigest()        
        hash_file = os.path.join('/scratch/shared/beegfs/htd/DATA/AutoAD/tmp', f'{hash_tag}.pre_sample.pickle')

        if os.path.exists(hash_file) and (not force_resample):
            self.all_clips = pickle.load(open(hash_file, 'rb'))
            print(f'load cached pre_sample_clips from {hash_file}')
        else:
            all_clips = []
            for movie_name in tqdm(self.movie_names, total=len(self.movie_names), desc='pre_sample_clips'):
                if movie_name not in self.anno_named:
                    continue
                anno = self.anno_named[movie_name]
                anno = anno.reset_index()  # make continuous index -- in case need to match GPT feature list
                anno['index'] = anno.index
                anno_someone = self.anno_someone[movie_name]
                anno_someone = anno_someone.reset_index()
                anno_someone['index'] = anno_someone.index

                v_feature = np.load(os.path.join(self.frame_feature_root, f'{movie_name}.npy'))  # this is 5 features per second
                feature_duration = v_feature.shape[0]
                movie_duration = anno.iloc[0]['movie_duration']
                avail_timestamps = math.floor(min(movie_duration * 5, feature_duration))  # 20/650 movie have inconsistent duration/feature_length/anno
                anno = anno[(anno['start'] <= avail_timestamps//5) & (anno['end'] <= avail_timestamps//5)]
                
                if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                    try:
                        charbank = torch.load(os.path.join(self.charbank_root, f'{movie_name}.charbank.pth.tar'))
                    except:
                        print(f"failed to load {os.path.join(self.charbank_root, f'{movie_name}.charbank.pth.tar')}")
                        continue

                    roles = []
                    # only use the first name
                    for i in charbank['roles']:
                        if i.split(' ')[0].endswith('.'):  # likely a prefix
                            try:
                                roles.append(rm_punct(i.split(' ')[1]))
                            except:
                                roles.append(i.split(' ')[0])  # maybe an initial
                        else:
                            roles.append(i.split(' ')[0])
                    roles = np.array(roles)
                    actors = np.array(charbank['names'])
                    charbank_active = charbank['cos'] > -1
                    if len(roles) > 0:
                        profile_topk_idx = torch.stack([topk_idx for topk_idx in charbank['top5_idx']], 0)
                    else:
                        profile_topk_idx = torch.empty(0)

                if self.mode in ['val', 'dev-val'] and self.real_mode != 'test':
                    row_idx = (len(anno) - self.num_clips - 1) // 2
                    all_clips.append(anno.iloc[row_idx: row_idx+self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.iloc[row_idx-2*self.num_clips: row_idx+2*self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.iloc[row_idx-4*self.num_clips: row_idx+4*self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.sort_values(by='start')) # take all
                    if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                        if len(roles) > 0:
                            active_char, active_actor, active_binary_map = self.get_charbank_string_from_df(
                                all_clips[-1], charbank_active, roles, actors)
                            all_clips[-1]['active_char'] = np.array(active_char)
                            all_clips[-1]['active_actor'] = np.array(active_actor)
                            all_clips[-1]['profile_topk_idx'] = [profile_topk_idx.tolist()] * self.num_clips
                            all_clips[-1]['active_binary_map'] = active_binary_map
                        else:
                            all_clips[-1]['active_char'] = ''
                            all_clips[-1]['active_actor'] = ''
                            all_clips[-1]['profile_topk_idx'] = ''
                            all_clips[-1]['active_binary_map'] = ''
                else:
                    for row_idx in range(0, len(anno), self.num_clips):
                        if row_idx + self.num_clips <= len(anno):
                            all_clips.append(anno.iloc[row_idx: row_idx + self.num_clips].sort_values(by='start'))
                        else:
                            all_clips.append(anno.iloc[len(anno) - self.num_clips :].sort_values(by='start'))
                        
                        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                            if len(roles) > 0:
                                active_char, active_actor, active_binary_map = self.get_charbank_string_from_df(
                                    all_clips[-1], charbank_active, roles, actors)
                                all_clips[-1]['active_char'] = np.array(active_char)
                                all_clips[-1]['active_actor'] = np.array(active_actor)
                                all_clips[-1]['profile_topk_idx'] = [profile_topk_idx.tolist()] * self.num_clips
                                all_clips[-1]['active_binary_map'] = active_binary_map
                            else:
                                all_clips[-1]['active_char'] = ''
                                all_clips[-1]['active_actor'] = ''
                                all_clips[-1]['active_profile'] = ''
                                all_clips[-1]['active_binary_map'] = ''
                
                if all_clips[-1].__len__() != self.num_clips:
                    print(f'warning: num_clips {all_clips[-1].__len__()} != {self.num_clips}')

            print(f'{identify_string} gets {len(all_clips)} clips')
            with open(hash_file, 'wb') as fobj:
                pickle.dump(all_clips, fobj)
            self.all_clips = all_clips


    def get_charbank_string_from_df(self, df, active_charbank_binary, roles, actors, FPS=5):
        """for a df with start and end timestamps, return a list of active char roles/actors"""
        assert len(roles) == len(actors) == active_charbank_binary.shape[0]
        assert len(roles) > 0
        active_char = []
        active_actor = []
        active_binary_map = []
        for start_idx, end_idx in zip(df['start'] * FPS, df['end'] * FPS):
            start_idx = max(round(start_idx), 0)
            end_idx = min(max(round(end_idx), start_idx+1), active_charbank_binary.shape[-1])
            active_char_map = active_charbank_binary[:, start_idx: end_idx].float().sum(-1) > 0.5
            active_binary_map.append(active_char_map.numpy())
            active_char.append(','.join(roles[active_char_map.numpy()]))
            active_actor.append(','.join(actors[active_char_map.numpy()]))
        return active_char, active_actor, active_binary_map


    def __getitem__(self, index):
        sampled_anno = self.all_clips[index]
        movie_name = sampled_anno.iloc[0]['movie']
        v_feature = np.load(os.path.join(self.frame_feature_root, f'{movie_name}.npy'))  # this is 5 features per second
        feature_duration = v_feature.shape[0]
        
        if self.mode in ['train']:
            random_shift = random.randint(0,4)
        else:
            random_shift = 0

        sentences = []
        clip_features = []
        start_times = []
        end_times = []
        all_active_char = []

        for row_idx, row in sampled_anno.iterrows():
            sent_named = row['sentence']
            word_list = sent_named.split()
            word_list = [i for i in word_list if '#' not in i]  # remove hash tag (remained from pose-processing)
            upper_word = [i for i in word_list if i.isupper() and len(i)>1]
            sent_someone = self.anno_someone[movie_name].loc[row_idx]['sentence']
            word_pairs = list(zip(sent_named.split(), sent_someone.split()))
            name_list = [p[0] for p in word_pairs if any([i in p[1] for i in ['SOMEONE', 'He', 'he', 'She', 'she']])]
            name_list = [n for n in name_list if n.isupper() or (n.capitalize()==n)]
            name_list = [n for n in name_list if n.upper() not in ['THE', 'A', 'SOMEONE', 'AS']]
            
            # if not all([n.isupper() or (n.capitalize()==n) for n in name_list]):
            #     import ipdb; ipdb.set_trace()  # it's common in LSMDC
            
            name_list = [rm_punct(i.upper()) for i in name_list]
            upper_word = [rm_punct(i) for i in upper_word]
            name_list = list(set(name_list + upper_word))
            name_list = [rm_punct(n) for n in name_list if n != '']
            if len(name_list) > 0:
                name_list_cap = [n.capitalize() for n in name_list]
                sentences.append(', '.join(name_list_cap))
                all_active_char.append(name_list_cap)
            else:
                sentences.append('unknown')
                all_active_char.append([])

            if self.lookahead == 2:  # slightly check ahead, shorter than 2x duration
                dur = row['end'] - row['start']
                ahead_idx_5fps = math.floor(max(0, row['start'] - dur) * 5)
                start_idx_5fps = ahead_idx_5fps
            else:
                start_idx_5fps = math.floor(row['start'] * 5)
            end_idx_5fps = math.ceil(row['end'] * 5)
            end_idx_5fps = min(end_idx_5fps, feature_duration)
            if start_idx_5fps+random_shift >= feature_duration:
                print(f'Error: index {start_idx_5fps+random_shift} to {end_idx_5fps-1} '
                      f'out of bounds for {feature_duration} features, continue ...')
                clip_feature = v_feature[[-1]*self.num_frames]
            else:
                clip_feature = v_feature[
                    np.linspace(start_idx_5fps+random_shift, end_idx_5fps - 1, self.num_frames, endpoint=False).astype(int)]
            
            clip_features.append(clip_feature)
            start_times.append((start_idx_5fps+random_shift)/5)
            # end_times.append(end_idx_5fps/5)
            end_times.append(row['end'])

        clip_features = torch.from_numpy(np.stack(clip_features, 0))  # num_clips, num_frames, C

        sentences = [' '+i.strip()+'.' for i in sentences]  # add leading space to match GPT2 pretraining
        tokens = self.tokenizer(sentences)['input_ids']
        tokens = [torch.LongTensor(i) for i in tokens]

        global_cast_list = sampled_anno["active_char"].iloc[0]
        global_cast_list = global_cast_list.split(',')
        global_cast_list = [i for i in global_cast_list if i!= '']
        global_cast_list = np.array(global_cast_list)
        binary_tgt = np.zeros((len(all_active_char), 10)).astype(int)
        num_C = len(global_cast_list)
        for i, item in enumerate(all_active_char):
            if len(item) == 0:
                continue
            active_char_array = np.array(item)
            try:
                binary_tgt[i, 0:num_C] = np.char.equal(global_cast_list[:,None], active_char_array[None,:]).astype(int).sum(-1) > 0
            except:
                print(f'error: {sampled_anno["active_char"].iloc[0]}')
                continue

        binary_tgt = torch.from_numpy(binary_tgt)

        video_padding_mask = torch.zeros(clip_features.shape[0:2]).long()
        return_dict =  {'video': clip_features, \
                'padding_mask': video_padding_mask, \
                'vid': movie_name, \
                'text': sentences, \
                'token': pad_sequence_to_length(tokens, length=36, value=self.tokenizer.eos_token_id), \
                'start': start_times, 
                'end': end_times, 
                'binary_tgt': binary_tgt,
                }
        charbank_return_dict = self.load_dense_charbank(sampled_anno, v_feature)
        return_dict.update(charbank_return_dict)
        del v_feature
        return return_dict


    def load_dense_charbank(self, sampled_anno, v_feature=None):
        return_dict = {}
        char_list = sampled_anno['active_char'].tolist()
        actor_list = sampled_anno['active_actor'].tolist()
        exampler_list = None
        if self.use_charbank in ['global-ce', 'global-cae'] and not all([i == '' for i in char_list]):
            profile_topk_idx = sampled_anno['profile_topk_idx'].iloc[0]
            all_exampler = []
            for topk_idx in profile_topk_idx:
                all_exampler.append(v_feature[np.array(topk_idx).astype(int), :].mean(0))
            all_exampler = np.stack(all_exampler, 0)
            active_binary_map = sampled_anno['active_binary_map'].tolist()
            exampler_list = []
            for binary_map in active_binary_map:
                exampler_list.append(all_exampler[np.array(binary_map)])

        char_list_with_prompt = []
        char_tokens = []
        char_identity_mask = []
        if exampler_list is None:
            exampler_list = [None] * len(char_list)
        if self.use_charbank in ['global-ce', 'global-cae']:
            exampler_feature_list = []
        
        for ch, ac, ex in zip(char_list, actor_list, exampler_list):
            if ch == '':
                char_list_with_prompt.append('possible characters: unknown.')
                char_item_token = self.tokenizer(['possible characters: unknown.'])['input_ids']
                char_item_token = char_item_token[0]
                char_identity = [-1] * len(char_item_token)
                char_tokens.append(char_item_token)
                char_identity_mask.append(char_identity)
                if self.use_charbank in [4, 41, 'global-ce', 'global-cae']:
                    exampler_feature_list.append(torch.empty(0, self.clip_dim))
            else:
                if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                    # method 2: give cast list
                    if self.use_charbank in ['global-char-act', 'global-cae']:
                        cast_list = [f'{i} played by {j}' for i,j in zip(ch.split(','), ac.split(','))]
                    else:
                        cast_list = [f'{i.strip()} <image>' for i,j in zip(ch.split(','), ac.split(','))]
                    cast_str = '; '.join(cast_list)
                    char_list_with_prompt.append(f'possible characters: {cast_str}.')
                    if self.use_charbank in [4, 41, 'global-ce', 'global-cae']:
                        exampler_feature_list.append(torch.from_numpy(ex))
                    
                    char_item_token = self.tokenizer(['possible characters: ']+cast_list)['input_ids']
                    char_identity = [[i-1]*len(tk) for i, tk in enumerate(char_item_token)]
                    char_item_token = [item for sublist in char_item_token for item in sublist]
                    char_identity = [item for sublist in char_identity for item in sublist]
                    char_tokens.append(char_item_token)
                    char_identity_mask.append(char_identity)
        
        # char_tokens = self.tokenizer(char_list_with_prompt)['input_ids']
        char_tokens = [torch.LongTensor(i) for i in char_tokens]
        char_identity_mask = [torch.LongTensor(i) for i in char_identity_mask]

        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
            max_len = max([len(tk) for tk in char_tokens])
            char_tokens = pad_sequence_to_length(char_tokens, length=max_len, value=self.tokenizer.eos_token_id)
            char_identity_mask = pad_sequence_to_length(char_identity_mask, length=max_len, value=-1)

            if self.use_charbank in ['global-ce', 'global-cae']:
                LEN = 10
                exampler_attn_mask = pad_sequence_to_length([torch.ones(i.shape[0]) for i in exampler_feature_list], 
                                                            length=LEN, value=0)
                exampler_feature_list = pad_sequence_to_length(exampler_feature_list, 
                                                               length=LEN, value=0)
                return_dict['exampler_attn_mask'] = exampler_attn_mask
                return_dict['exampler_feature'] = exampler_feature_list

        return_dict['char_tokens'] = char_tokens
        return_dict['char_text'] = char_list_with_prompt
        return_dict['char_identity'] = char_identity_mask
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['padding_mask'] = pad_sequence([sample['padding_mask'] for sample in batch], batch_first=True, padding_value=1.0)
        out_batch['text'] = [sample['text'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['vid'] = [sample['vid'] for sample in batch]
        out_batch['token'] = [sample['token'] for sample in batch]
        out_batch['binary_tgt'] = default_collate([sample['binary_tgt'] for sample in batch])
        if 'char_tokens' in batch[0]:
            out_batch['char_tokens'] = pad_sequence_by_dim([sample['char_tokens'] for sample in batch], dim=1, value=EOS_TOKEN_ID)
            out_batch['char_text'] = [sample['char_text'] for sample in batch]
            out_batch['char_identity'] = pad_sequence_by_dim([sample['char_identity'] for sample in batch], dim=1, value=-1)
        if 'exampler_feature' in batch[0]:
            out_batch['exampler_feature'] = default_collate([sample['exampler_feature'] for sample in batch])
            out_batch['exampler_attn_mask'] = default_collate([sample['exampler_attn_mask'] for sample in batch])
        return out_batch 

    def __len__(self):
        return self.dataset_length
    


class MovieNet_NameLoader():
    def __init__(self,
                 mode='train', tokenizer=None, 
                 num_frames=8, num_clips=4,
                 use_charbank=0,
                 lookahead=0,
                 force_resample=False,
                 clip_version='L14',
                 **kwargs):
        if len(kwargs):
            print(f'MovieNet_NameLoader: {kwargs} not used by dataset')
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.use_charbank = use_charbank
        self.mode = mode
        movienet_anno = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/movienet/movienet_face_anno.csv')
        # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal'
        # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_2023mar'
        # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_2023jun'
        # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_v31_2023jun'
        self.charbank_root = '/scratch/shared/beegfs/htd/MAD/charbank_cos_top10_cal_2023jul'
        self.movienet_charbank_root = '/scratch/shared/beegfs/htd/MovieNet/charbank_cos_top10'
        self.clip_version = clip_version
        if clip_version == 'B32':
            self.clip_dim = 512
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_frames_features_5fps'
            self.feature_path = '/scratch/shared/beegfs/htd/DATA/MovieNet/keyframe_feat/openai-clip-vit-b-32'
            raise NotImplementedError(f"movienet_feature_path not processed")
        elif clip_version == 'L14':
            self.clip_dim = 768
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_L14_frames_features_5fps'
            self.feature_path = '/scratch/shared/beegfs/htd/DATA/MovieNet/keyframe_feat/openai-clip-vit-l-14'
            self.movienet_frame_feature_root = self.feature_path

        # check char bank
        VER = 550
        print(f'Warning: using MovieNet-{VER} version, character bank from {self.charbank_root} and {self.movienet_charbank_root}')
        movienet_anno = self.get_movienet_subset(movienet_anno, version=VER)
        drop_movie_id = ['tt0455824', 'tt6644200', 'tt0077402', 'tt1139797',
                         'tt0063442', 'tt0079470', 'tt0089881', 'tt0095765', 
                         'tt0374546', 'tt0844347', 'tt1611840', 'tt2011351', 
                         'tt2115388', 'tt5593416', 'tt6121428']
        movienet_anno = movienet_anno[~movienet_anno['movie_id'].isin(drop_movie_id)].copy()
        movienet_anno = movienet_anno[~movienet_anno['pid'].isna()].copy()
        if mode != "train":
            mad_eval_intersect = ['tt0286106', 'tt0330373', 'tt1124035', 'tt1707386']
            movienet_anno = movienet_anno[movienet_anno['movie_id'].isin(mad_eval_intersect)].copy()
        self.movienet_anno = movienet_anno
        print(f"MovieNet_NameLoader version {VER} {mode} uses {len(self.movienet_anno.movie_id.unique())} movies")

        # char id to name mapping
        with open(os.path.join("/work/htd/Desktop_tmp/AutoMad/data_post_proc/MAD_charbank_2023mar.json")) as fobj:
            charbank_dict = json.load(fobj)
        with open(os.path.join("/work/htd/Desktop_tmp/AutoMad/char_bank/MovieNet_charbank_300.json")) as fobj:
            movienet_charbank_dict = json.load(fobj)
        if VER == 550:
            with open(os.path.join("/work/htd/Desktop_tmp/AutoMad/char_bank/MovieNet_charbank_patch_148.json")) as fobj:
                patch_dict = json.load(fobj)
            movienet_charbank_dict = {**movienet_charbank_dict, **patch_dict}
        self.char_mapping_per_movie = {}
        movienet_movie_ids = sorted(self.movienet_anno.movie_id.unique().tolist())

        for movie_id in movienet_movie_ids:
            if movie_id in self.imdb_to_mad_name:
                charinfo = charbank_dict[self.imdb_to_mad_name.get(movie_id)]
            else:
                charinfo = movienet_charbank_dict[movie_id]
            char_id_to_name = {item['id']: item['name'] for item in charinfo}
            self.char_mapping_per_movie[movie_id] = char_id_to_name

        # pre-load exemplars
        drop_id = []
        hash_exemplars = hashlib.sha256(json.dumps(movienet_movie_ids).encode()).hexdigest()        
        hash_file_e = os.path.join('/scratch/shared/beegfs/htd/AutoAD/tmp', f'{hash_exemplars}.exemplar.pickle')
        if os.path.exists(hash_file_e) and (not force_resample):
            self.exemplar_per_movie = pickle.load(open(hash_file_e, 'rb'))
        else:
            self.exemplar_per_movie = {}
            for movie_id in tqdm(movienet_movie_ids, desc="exemplars", leave=False):
                if movie_id in self.imdb_to_mad_name:
                    charbank = torch.load(os.path.join(self.charbank_root, f"{self.imdb_to_mad_name.get(movie_id)}.charbank.pth.tar"))
                    v_feature = np.load(os.path.join(self.frame_feature_root, f'{self.imdb_to_mad_name.get(movie_id)}.npy'))  # this is 5 features per second
                else:
                    charbank = torch.load(os.path.join(self.movienet_charbank_root, f"{movie_id}.charbank.pth.tar"))
                    v_feature = np.load(os.path.join(self.movienet_frame_feature_root, f'{movie_id}.npy'))  # this is 5 features per second
                try:
                    indexes = torch.stack(charbank['top5_idx'], 0).long()
                except:
                    print(f"{movie_id} has an empty charbank")
                    drop_id.append(movie_id)
                    continue
                try:
                    v_exem = v_feature[indexes.view(-1).numpy(), :].reshape((-1, 5, v_feature.shape[-1])).mean(1)
                except:
                    import ipdb; ipdb.set_trace()
                if torch.all(indexes[:,0] == -1):
                    print(f"{movie_id} does not have valid char bank")
                    drop_id.append(movie_id)
                    continue
                self.exemplar_per_movie[movie_id] = (v_exem, indexes[:,0] != -1)
            pickle.dump(self.exemplar_per_movie, open(hash_file_e, 'wb'))

        if len(drop_id):
            print(f"DropID={drop_id}")
            self.movienet_anno = self.movienet_anno[~self.movienet_anno['movie_id'].isin(drop_id)].copy()

        # split
        # assert mode == 'train'
        # mode_conversion = {'test': 'val'}
        # mode_ = mode_conversion.get(mode, mode)
        # if mode_ == 'train':
        #     self.movie_names = LSMDC_TRAIN
        # else:
        #     self.movie_names = LSMDC_VAL
        self.movie_names = sorted(self.movienet_anno.movie_id.unique().tolist())

        # pre-sample neighbours
        movienet_anno_per_movie = dict(tuple(self.movienet_anno.groupby('movie_id')))
        identify_string = f'MovieNet_Name_{self.mode}_{self.num_frames}_{self.use_charbank}'
        hash_tag = hashlib.sha256((identify_string+json.dumps(self.movie_names)).encode()).hexdigest()        
        hash_file = os.path.join('/scratch/shared/beegfs/htd/AutoAD/tmp', f'{hash_tag}.pre_sample.pickle')
        if os.path.exists(hash_file) and (not force_resample):
            self.all_clips = pickle.load(open(hash_file, 'rb'))
            print(f'load cached pre_sample_clips from {hash_file}')
        else:
            all_clips = []
            for movie_id, df in tqdm(movienet_anno_per_movie.items(), desc="pre-sample neighbours", leave=False):
                if movie_id in self.imdb_to_mad_name:
                    charbank = torch.load(os.path.join(self.charbank_root, f"{self.imdb_to_mad_name.get(movie_id)}.charbank.pth.tar"))
                else:
                    charbank = torch.load(os.path.join(self.movienet_charbank_root, f"{movie_id}.charbank.pth.tar"))
                for row_idx in range(0, len(df), self.num_frames):
                    if row_idx + self.num_frames <= len(df):
                        rows = df.iloc[row_idx: row_idx + self.num_frames]
                    else:
                        rows = df.iloc[len(df) - self.num_frames :]
                    rows = rows.copy()
                    pids = list(set(rows.pid.tolist()))
                    names = [self.char_mapping_per_movie[movie_id].get(i) for i in pids]
                    names = [i for i in names if i is not None]
                    if len(names) == 0:
                        print(f"{movie_id}-row{row_idx}: Names are {names}")
                        continue
                    charbank_names = charbank['names']
                    rows['char_names'] = '|'.join(names)
                    rows['charbank_names'] = '|'.join(charbank_names)
                    all_clips.append(rows)
            print(f'{identify_string} gets {len(all_clips)} clips')
            with open(hash_file, 'wb') as fobj:
                pickle.dump(all_clips, fobj)
            self.all_clips = all_clips

    def get_movienet_subset(self, movienet_anno, version=100):
        assert version in [100, 400, 550], f"{version=} is not supported"
        mad_imdb_info = pd.read_csv('/scratch/shared/beegfs/htd/DATA/MAD/mad_imdb_info.csv')
        imdb_to_mad_name = dict(zip(mad_imdb_info.imdb.tolist(), mad_imdb_info.movie_name.tolist()))
        self.imdb_to_mad_name = imdb_to_mad_name
        intersection_imdb = set(mad_imdb_info.imdb.tolist()).intersection(set(movienet_anno.movie_id.unique().tolist()))
        val_imdb = mad_imdb_info[mad_imdb_info['movie_name'].isin(MAD_EVAL)].imdb.tolist()  # LSMDC_VAL or MAD_EVAL

        if version == 100:
            assert self.mode == 'train'
            intersection_imdb = intersection_imdb - set(val_imdb)
            movienet_anno = movienet_anno[movienet_anno['movie_id'].isin(intersection_imdb)].copy()
            movienet_anno = movienet_anno[~movienet_anno.pid.isna()].copy()
            tmp_movie_imdb = movienet_anno['movie_id'].unique().tolist() # exist?
            tmp_movie_imdb = [i for i in tmp_movie_imdb if os.path.exists(os.path.join(self.feature_path, f"{i}.npy"))]
            movienet_anno = movienet_anno[movienet_anno['movie_id'].isin(tmp_movie_imdb)].copy()
        elif version in [400, 550]:
            processed_movienet_imdb = glob("/scratch/shared/beegfs/htd/MovieNet/charbank_cos_top10/*.pth.tar")
            processed_movienet_imdb = sorted([os.path.basename(i).split('.')[0] for i in processed_movienet_imdb])
            exist_mad_imdb = [i for i in sorted(list(intersection_imdb)) if os.path.exists(os.path.join(self.feature_path, f"{i}.npy"))]
            if self.mode == 'train':
                intersection_imdb = set(exist_mad_imdb).union(set(processed_movienet_imdb)) - set(val_imdb)
            if version == 400:  # ensure it's 400 version (not including the 148 patch)
                with open(os.path.join("/work/htd/Desktop_tmp/AutoMad/char_bank/MovieNet_charbank_patch_148.json")) as fobj:
                    patch_dict = json.load(fobj)
                intersection_imdb = intersection_imdb - set(patch_dict.keys())
            self.processed_movienet_imdb = processed_movienet_imdb
            movienet_anno = movienet_anno[movienet_anno['movie_id'].isin(intersection_imdb)].copy()
        return movienet_anno


    def __getitem__(self, index):
        rows = self.all_clips[index]
        movie_id = rows.iloc[0]['movie_id']
        if movie_id in self.imdb_to_mad_name:
            movie_name = self.imdb_to_mad_name.get(movie_id)
        else:
            movie_name = movie_id
        assert os.path.exists(os.path.join(self.feature_path, movie_id + '.npy'))
        feature = np.load(os.path.join(self.feature_path, movie_id + '.npy'))
        meta = open(os.path.join(self.feature_path, movie_id + '.txt')).readlines()
        meta = [i.strip() for i in meta]

        # rep_pool = [[1,1,6], [1,2,5], [1,3,4], [1,4,3], [1,5,2], [1,6,1],
        #             [2,1,5], [2,2,4], [2,3,3], [2,4,2], [2,5,1],
        #             [3,1,4], [3,2,3], [3,3,2], [3,4,1],
        #             [4,1,3], [4,2,2], [4,3,1],
        #             [5,1,2], [5,2,1],
        #             [6,1,1]]
        # rep_count = random.choice(rep_pool)
        # assert sum(rep_count) == self.num_frames

        features = []
        for idx, row in rows.iterrows():
            tgt_filename = f"shot_{row['shot_idx']:04d}_img_{row['img_idx']}.jpg"
            try:
                tgt_feature_idx = meta.index(tgt_filename)
            except:
                print(f"{movie_id} {tgt_filename} does not exist")
                tgt_feature_idx = -1
            tgt_feature = feature[tgt_feature_idx]
            # features.append(torch.from_numpy(tgt_feature)[None,:].repeat(rep_count[idx],1))
            features.append(torch.from_numpy(tgt_feature)[None,:])
        clip_features = torch.cat(features, 0).float()
        # group into shot 3 frames -> 8 frames
        # charbank = torch.load(os.path.join(self.charbank_root, f"{movie_name}.charbank.pth.tar"))
        
        LEN = 10
        exemplar, exemplar_mask = self.exemplar_per_movie[movie_id]
        exemplar = torch.from_numpy(exemplar).float()
        exemplar_mask = exemplar_mask.float()
        if exemplar_mask.shape[0] != LEN:
            exemplar_mask = torch.cat((exemplar_mask, torch.zeros(LEN - exemplar_mask.shape[0])), 0)
        if exemplar.shape[0] != LEN:
            exemplar = torch.cat((exemplar, torch.zeros(LEN - exemplar.shape[0], exemplar.shape[-1])), 0)
        assert exemplar.shape[0] == LEN
        assert exemplar_mask.shape[0] == LEN

        all_active_char = np.array(rows.iloc[0]['char_names'].split('|'))
        global_cast_list = np.array(rows.iloc[0]['charbank_names'].split('|'))
        binary_tgt = np.char.equal(global_cast_list[:,None], all_active_char[None,:]).astype(int).sum(-1) > 0
        binary_tgt = torch.from_numpy(binary_tgt).long()
        if binary_tgt.shape[0] != LEN:
            binary_tgt = torch.cat((binary_tgt, torch.zeros(LEN - binary_tgt.shape[0]).long()))

        return_dict =  {'video': clip_features[None,:], \
                'padding_mask': torch.zeros_like(clip_features[...,0])[None,:], \
                'vid': movie_name, \
                'text': '', \
                'token': torch.zeros(1,36), \
                'start': [0], 
                'end': [1], 
                'binary_tgt': binary_tgt[None,:],
                'exampler_feature': exemplar[None,:],
                'exampler_attn_mask': exemplar_mask[None,:]
                }
        return return_dict
        
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['padding_mask'] = pad_sequence([sample['padding_mask'] for sample in batch], batch_first=True, padding_value=1.0)
        out_batch['text'] = [sample['text'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['vid'] = [sample['vid'] for sample in batch]
        out_batch['token'] = [sample['token'] for sample in batch]
        out_batch['binary_tgt'] = default_collate([sample['binary_tgt'] for sample in batch])
        if 'char_tokens' in batch[0]:
            out_batch['char_tokens'] = pad_sequence_by_dim([sample['char_tokens'] for sample in batch], dim=1, value=EOS_TOKEN_ID)
            out_batch['char_text'] = [sample['char_text'] for sample in batch]
            out_batch['char_identity'] = pad_sequence_by_dim([sample['char_identity'] for sample in batch], dim=1, value=-1)
        if 'exampler_feature' in batch[0]:
            out_batch['exampler_feature'] = default_collate([sample['exampler_feature'] for sample in batch])
            out_batch['exampler_attn_mask'] = default_collate([sample['exampler_attn_mask'] for sample in batch])
        return out_batch
    
    @staticmethod
    def collate_fn_aggregate(batch):
        """only used when merging with other dataset -- less key-value pairs"""
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['padding_mask'] = pad_sequence([sample['padding_mask'] for sample in batch], batch_first=True, padding_value=1.0)
        out_batch['text'] = [sample['text'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['vid'] = [sample['vid'] for sample in batch]
        out_batch['token'] = [sample['token'] for sample in batch]
        out_batch['binary_tgt'] = default_collate([sample['binary_tgt'] for sample in batch])
        out_batch['exampler_feature'] = default_collate([sample['exampler_feature'] for sample in batch])
        out_batch['exampler_attn_mask'] = default_collate([sample['exampler_attn_mask'] for sample in batch])
        return out_batch

    def __len__(self):
        return len(self.all_clips)



class MAD_NameLoader():
    def __init__(self,
                 mode='train', tokenizer=None, 
                 num_frames=8, num_clips=4,
                 use_charbank=0,
                 lookahead=0,
                 force_resample=False,
                 clip_version='B32',
                 exclude_movienet=False,
                 **kwargs):
        if len(kwargs):
            print(f'MAD_NameLoader: {kwargs} not used by dataset')

        assert mode == 'train'
        ner = pd.read_csv('/work/maxbain/datasets/MAD_Language_Grounding/audiovault_audios_MAD_training_set/'
                               'asr/ad_segments_madv3/ad-mad-v3_NSSD_jaesung_0p95thresh_minAD95_pronthresh1p0-NER.csv')
        self.ner = dict(tuple(ner.groupby('movie')))
        self.ner = {str(k):v for k,v in self.ner.items()}
        
        version_card = {'lsmdc': '/scratch/shared/beegfs/htd/DATA/MAD/MAD_anno_dict_pd.pkl',  # someone
                        'lsmdc_named': '/scratch/shared/beegfs/htd/DATA/MAD/MAD_named_anno_dict_pd.pkl',  # named
                        'v3_named_unchanged': '/work/htd/Desktop_tmp/AutoMad/data_post_proc/mad_whisper_v3_unchanged.pkl', 
                        }
        self.anno_named = pickle.load(open(version_card['v3_named_unchanged'], 'rb'))
        self.movie_id_to_anno = {key.split('_')[0]: key for key in self.anno_named.keys()}
        self.lookahead = lookahead
        self.tokenizer = tokenizer
        self.answer_prompt = ' Characters in <video>:'
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.mode = mode
        self.movie_names = json.load(open('/scratch/shared/beegfs/htd/DATA/MAD/MAD_movie_name_by_split.json', 'r'))['train']
        self.exclude_movienet = exclude_movienet

        if exclude_movienet:
            pre_length = len(self.movie_names)
            movienet_anno = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/movienet/movienet_face_anno.csv')
            mvn_id = set(movienet_anno['movie_id'].unique().tolist())
            mad_imdb_info = pd.read_csv("/scratch/shared/beegfs/htd/DATA/MAD/mad_imdb_info.csv")
            mad_name_to_imdb = dict(zip(mad_imdb_info['movie_name'], mad_imdb_info['imdb']))
            tmp_movie_names = []
            for name in self.movie_names:
                if mad_name_to_imdb.get(name) in mvn_id:
                    continue
                tmp_movie_names.append(name)
            self.movie_names = tmp_movie_names
            print(f"Excluding MovieNet movies reduces movies from {pre_length} to {len(self.movie_names)}")

        # CharBank
        self.use_charbank = use_charbank
        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']: # give global charbank
            # TODO: understand the diff of later versions
            # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal'
            # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_2023mar'
            # self.charbank_root = '/scratch/shared/beegfs/htd/DATA/MAD/charbank_cos_top10_cal_2023jun'
            self.charbank_root = '/scratch/shared/beegfs/htd/MAD/charbank_cos_top10_cal_2023jul'

        self.clip_version = clip_version
        if clip_version == 'B32':
            self.dim = 512
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_frames_features_5fps'
        elif clip_version == 'L14':
            self.dim = 768
            self.frame_feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_L14_frames_features_5fps'

        self.pre_sample_clips(force_resample=force_resample)  # 'val' in mode_
        self.dataset_length = len(self.all_clips)
        print(f'Loaded {mode}-set from {self.frame_feature_root}')


    def pre_sample_clips(self, force_resample=False):
        # continuous clips from movies
        identify_string = f'MAD_Name_{self.mode}_{self.num_clips}_{self.use_charbank}'
        hash_tag = hashlib.sha256((identify_string+json.dumps(self.movie_names)).encode()).hexdigest()        
        hash_file = os.path.join('/scratch/shared/beegfs/htd/DATA/AutoAD/tmp', f'{hash_tag}.pre_sample.pickle')

        if os.path.exists(hash_file) and (not force_resample):
            self.all_clips = pickle.load(open(hash_file, 'rb'))
            print(f'load cached pre_sample_clips from {hash_file}')
        else:
            all_clips = []
            # load intro-outro pts (for train val)
            intro_outro = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/char_bank/MAD_intro_outro.csv')

            for movie_name in tqdm(self.movie_names, total=len(self.movie_names), desc='pre_sample_clips'):
                if movie_name not in self.anno_named:
                    continue
                anno = self.anno_named[movie_name]
                anno = anno.reset_index()  # make continuous index -- in case need to match GPT feature list
                anno['index'] = anno.index

                ner = self.ner[movie_name]
                assert len(ner) == len(anno)
                anno['ner'] = ner['ner']

                v_feature = np.load(os.path.join(self.frame_feature_root, f'{movie_name}.npy'))  # this is 5 features per second
                feature_duration = v_feature.shape[0]
                movie_duration = anno.iloc[0]['movie_duration']
                avail_timestamps = math.floor(min(movie_duration * 5, feature_duration))  # 20/650 movie have inconsistent duration/feature_length/anno
                anno = anno[(anno['start'] <= avail_timestamps//5) & (anno['end'] <= avail_timestamps//5)]
                anno = anno[anno['sentence'].str.strip() != ""]
                
                intro = intro_outro[intro_outro["vid"] == movie_name]["start_pts"].item()
                outro = intro_outro[intro_outro["vid"] == movie_name]["end_pts"].item()
                anno = anno[(anno['start'] >= intro//5) & (anno['end'] <= outro//5)]

                if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                    if not os.path.exists(os.path.join(self.charbank_root, f'{movie_name}.charbank.pth.tar')):
                        print(f"{os.path.join(self.charbank_root, f'{movie_name}.charbank.pth.tar')} not exists")
                        continue
                    charbank = torch.load(os.path.join(self.charbank_root, f'{movie_name}.charbank.pth.tar'))
                    roles = []
                    # only use the first name
                    for i in charbank['roles']:
                        if i.split(' ')[0].endswith('.'):  # likely a prefix
                            try:
                                roles.append(rm_punct(i.split(' ')[1]))
                            except:
                                roles.append(i.split(' ')[0])  # maybe an initial
                        else:
                            roles.append(i.split(' ')[0])
                    roles = np.array(roles)
                    actors = np.array(charbank['names'])
                    charbank_active = charbank['cos'] > -1
                    if len(roles) > 0:
                        profile_topk_idx = torch.stack([topk_idx for topk_idx in charbank['top5_idx']], 0)
                    else:
                        profile_topk_idx = torch.empty(0)

                if len(roles) == 0 or all([i=='' for i in roles]):
                    import ipdb; ipdb.set_trace()
                    continue

                if self.mode in ['val', 'dev-val']:
                    row_idx = (len(anno) - self.num_clips - 1) // 2
                    all_clips.append(anno.iloc[row_idx: row_idx+self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.iloc[row_idx-2*self.num_clips: row_idx+2*self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.iloc[row_idx-4*self.num_clips: row_idx+4*self.num_clips].sort_values(by='start'))
                    # all_clips.append(anno.sort_values(by='start')) # take all
                    if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                        if len(roles) > 0:
                            active_char, active_actor, active_binary_map = self.get_charbank_string_from_df(
                                all_clips[-1], charbank_active, roles, actors)
                            all_clips[-1]['active_char'] = np.array(active_char)
                            all_clips[-1]['active_actor'] = np.array(active_actor)
                            all_clips[-1]['profile_topk_idx'] = [profile_topk_idx.tolist()] * self.num_clips
                            all_clips[-1]['active_binary_map'] = active_binary_map
                        else:
                            all_clips[-1]['active_char'] = ''
                            all_clips[-1]['active_actor'] = ''
                            all_clips[-1]['profile_topk_idx'] = ''
                            all_clips[-1]['active_binary_map'] = ''
                else:
                    for row_idx in range(0, len(anno), self.num_clips):
                        if row_idx + self.num_clips <= len(anno):
                            all_clips.append(anno.iloc[row_idx: row_idx + self.num_clips].sort_values(by='start'))
                        else:
                            all_clips.append(anno.iloc[len(anno) - self.num_clips :].sort_values(by='start'))
                        
                        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                            if len(roles) > 0:
                                active_char, active_actor, active_binary_map = self.get_charbank_string_from_df(
                                    all_clips[-1], charbank_active, roles, actors)
                                all_clips[-1]['active_char'] = np.array(active_char)
                                all_clips[-1]['active_actor'] = np.array(active_actor)
                                all_clips[-1]['profile_topk_idx'] = [profile_topk_idx.tolist()] * self.num_clips
                                all_clips[-1]['active_binary_map'] = active_binary_map
                            else:
                                all_clips[-1]['active_char'] = ''
                                all_clips[-1]['active_actor'] = ''
                                all_clips[-1]['active_profile'] = ''
                                all_clips[-1]['active_binary_map'] = ''
                
                if all_clips[-1].__len__() != self.num_clips:
                    print(f'warning: num_clips {all_clips[-1].__len__()} != {self.num_clips}')

            all_clips = [clip for clip in all_clips if clip.iloc[0]["active_char"] != '']
            print(f'{identify_string} gets {len(all_clips)} clips')
            with open(hash_file, 'wb') as fobj:
                pickle.dump(all_clips, fobj)
            self.all_clips = all_clips


    def get_charbank_string_from_df(self, df, active_charbank_binary, roles, actors, FPS=5):
        """for a df with start and end timestamps, return a list of active char roles/actors"""
        assert len(roles) == len(actors) == active_charbank_binary.shape[0]
        assert len(roles) > 0
        active_char = []
        active_actor = []
        active_binary_map = []
        for start_idx, end_idx in zip(df['start'] * FPS, df['end'] * FPS):
            start_idx = max(round(start_idx), 0)
            end_idx = min(max(round(end_idx), start_idx+1), active_charbank_binary.shape[-1])
            active_char_map = active_charbank_binary[:, start_idx: end_idx].float().sum(-1) > 0.5
            active_binary_map.append(active_char_map.numpy())
            active_char.append(','.join(roles[active_char_map.numpy()]))
            active_actor.append(','.join(actors[active_char_map.numpy()]))
        return active_char, active_actor, active_binary_map


    def __getitem__(self, index):
        sampled_anno = self.all_clips[index]
        movie_name = sampled_anno.iloc[0]['movie']
        v_feature = np.load(os.path.join(self.frame_feature_root, f'{movie_name}.npy'))  # this is 5 features per second
        feature_duration = v_feature.shape[0]
        
        if self.mode in ['train']:
            random_shift = random.randint(0,4)
        else:
            random_shift = 0

        sentences = []
        clip_features = []
        start_times = []
        end_times = []
        all_active_char = []

        for row_idx, row in sampled_anno.iterrows():
            sent_named = row['sentence']
            try:
                ner_list = ast.literal_eval(row['ner'])
            except:
                ner_list = []
            ner_list = [i['word'] for i in ner_list if i['entity_group']=='PER']
            new_l = []
            for i in ner_list:
                if ' ' in i:
                    if ('.' in i.split(' ')[0]):
                        try:
                            firstname = i.split(' ')[1]
                        except:
                            firstname = i.split(' ')[0]
                    else:
                        firstname = i.split(' ')[0]
                else:
                    firstname = i
                new_l.append(firstname)
            ner_list = new_l
            name_list = [n.strip() for n in ner_list if n != '']

            # also add capitalized words as candicate
            word_list = sent_named.split()
            cap_word = [i for i in word_list if i==i.capitalize() and len(i)>1]
            cap_word = [rm_punct(n).strip() for n in cap_word if n not in ['He','She','The','They','Them','It','Its','As','And','But','Or','How','This','That','In']]
            name_list = list(set(name_list + cap_word))

            # word_list = sent_named.split()
            # word_list = [i for i in word_list if '#' not in i]  # remove hash tag (remained from pose-processing)
            # upper_word = [i for i in word_list if i.isupper() and len(i)>1]
            # sent_someone = self.anno_someone[movie_name].loc[row_idx]['sentence']
            # word_pairs = list(zip(sent_named.split(), sent_someone.split()))
            # name_list = [p[0] for p in word_pairs if 'SOMEONE' in p[1]]
            # name_list = [n for n in name_list if n.isupper() or (n.capitalize()==n)]
            # name_list = [n for n in name_list if n.upper() not in ['THE', 'A', 'SOMEONE', 'AS']]
            # name_list = [rm_punct(i.upper()) for i in name_list]
            # upper_word = [rm_punct(i) for i in upper_word]
            # name_list = list(set(name_list + upper_word))
            # name_list = [rm_punct(n) for n in name_list if n != '']
            if len(name_list) > 0:
                name_list_cap = [n.capitalize() for n in name_list]
                sentences.append(', '.join(name_list_cap))
                all_active_char.append(name_list_cap)
            else:
                sentences.append('unknown')
                all_active_char.append([])

            if self.lookahead == 2:  # slightly check ahead, shorter than 2x duration
                dur = row['end'] - row['start']
                if dur < 2:
                    expansion = 2 - dur
                else:
                    expansion = 0
                ahead_idx_5fps = math.floor(max(0, row['start'] - expansion) * 5)
                start_idx_5fps = ahead_idx_5fps
            else:
                start_idx_5fps = math.floor(row['start'] * 5)
            end_idx_5fps = math.ceil(row['end'] * 5)
            end_idx_5fps = min(end_idx_5fps, feature_duration)
            if start_idx_5fps+random_shift >= feature_duration:
                print(f'Error: index {start_idx_5fps+random_shift} to {end_idx_5fps-1} '
                      f'out of bounds for {feature_duration} features, continue ...')
                clip_feature = v_feature[[-1]*self.num_frames]
            else:
                clip_feature = v_feature[
                    np.linspace(start_idx_5fps+random_shift, end_idx_5fps - 1, self.num_frames, endpoint=False).astype(int)]
            
            clip_features.append(clip_feature)
            start_times.append((start_idx_5fps+random_shift)/5)
            # end_times.append(end_idx_5fps/5)
            end_times.append(row['end'])

        clip_features = torch.from_numpy(np.stack(clip_features, 0))  # num_clips, num_frames, C

        sentences = [' '+i.strip()+'.' for i in sentences]  # add leading space to match GPT2 pretraining
        tokens = self.tokenizer(sentences)['input_ids']
        tokens = [torch.LongTensor(i) for i in tokens]

        global_cast_list = sampled_anno["active_char"].iloc[0]
        global_cast_list = global_cast_list.split(',')
        global_cast_list = [i for i in global_cast_list if i!= '']
        global_cast_list = [i.capitalize() for i in global_cast_list]
        global_cast_list = np.array(global_cast_list).astype(str)
        binary_tgt = np.zeros((len(all_active_char), 10)).astype(int)
        num_C = len(global_cast_list)
        for i, item in enumerate(all_active_char):
            if len(item) == 0:
                continue
            item = [i.capitalize() for i in item]
            active_char_array = np.array(item).astype(str)
            try:
                binary_tgt[i, 0:num_C] = np.char.equal(global_cast_list[:,None], active_char_array[None,:]).astype(int).sum(-1) > 0
            except:
                print(f'error: {sampled_anno["active_char"].iloc[0]}')
                continue

        binary_tgt = torch.from_numpy(binary_tgt)

        video_padding_mask = torch.zeros(clip_features.shape[0:2]).long()
        return_dict =  {'video': clip_features, \
                'padding_mask': video_padding_mask, \
                'vid': movie_name, \
                'text': sentences, \
                'token': pad_sequence_to_length(tokens, length=36, value=self.tokenizer.eos_token_id), \
                'start': start_times, 
                'end': end_times, 
                'binary_tgt': binary_tgt,
                }
        
        charbank_return_dict = self.load_dense_charbank(sampled_anno, v_feature)
        return_dict.update(charbank_return_dict)
        del v_feature
        return return_dict


    def load_dense_charbank(self, sampled_anno, v_feature=None):
        return_dict = {}
        char_list = sampled_anno['active_char'].tolist()
        actor_list = sampled_anno['active_actor'].tolist()
        exampler_list = None
        if self.use_charbank in ['global-ce', 'global-cae'] and not all([i == '' for i in char_list]):
            profile_topk_idx = sampled_anno['profile_topk_idx'].iloc[0]
            all_exampler = []
            for topk_idx in profile_topk_idx:
                all_exampler.append(v_feature[np.array(topk_idx).astype(int), :].mean(0))
            all_exampler = np.stack(all_exampler, 0)
            active_binary_map = sampled_anno['active_binary_map'].tolist()
            exampler_list = []
            for binary_map in active_binary_map:
                exampler_list.append(all_exampler[np.array(binary_map)])

        char_list_with_prompt = []
        char_tokens = []
        char_identity_mask = []
        if exampler_list is None:
            exampler_list = [None] * len(char_list)
        if self.use_charbank in ['global-ce', 'global-cae']:
            exampler_feature_list = []
        
        for ch, ac, ex in zip(char_list, actor_list, exampler_list):
            if ch == '':
                import ipdb; ipdb.set_trace()
                char_list_with_prompt.append('possible characters: unknown.')
                char_item_token = self.tokenizer(['possible characters: unknown.'])['input_ids']
                char_item_token = char_item_token[0]
                char_identity = [-1] * len(char_item_token)
                char_tokens.append(char_item_token)
                char_identity_mask.append(char_identity)
                if self.use_charbank in [4, 41, 'global-ce', 'global-cae']:
                    exampler_feature_list.append(torch.empty(0, self.dim))
            else:
                if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
                    # method 2: give cast list
                    if self.use_charbank in ['global-char-act', 'global-cae']:
                        cast_list = [f'{i} played by {j}' for i,j in zip(ch.split(','), ac.split(','))]
                    else:
                        cast_list = [f'{i.strip()} <image>' for i,j in zip(ch.split(','), ac.split(','))]
                    cast_str = '; '.join(cast_list)
                    char_list_with_prompt.append(f'possible characters: {cast_str}.')
                    if self.use_charbank in [4, 41, 'global-ce', 'global-cae']:
                        exampler_feature_list.append(torch.from_numpy(ex))
                    
                    char_item_token = self.tokenizer(['possible characters: ']+cast_list)['input_ids']
                    char_identity = [[i-1]*len(tk) for i, tk in enumerate(char_item_token)]
                    char_item_token = [item for sublist in char_item_token for item in sublist]
                    char_identity = [item for sublist in char_identity for item in sublist]
                    char_tokens.append(char_item_token)
                    char_identity_mask.append(char_identity)
        
        # char_tokens = self.tokenizer(char_list_with_prompt)['input_ids']
        char_tokens = [torch.LongTensor(i) for i in char_tokens]
        char_identity_mask = [torch.LongTensor(i) for i in char_identity_mask]

        if self.use_charbank in ['global-char', 'global-char-act', 'global-ce', 'global-cae']:
            max_len = max([len(tk) for tk in char_tokens])
            char_tokens = pad_sequence_to_length(char_tokens, length=max_len, value=self.tokenizer.eos_token_id)
            char_identity_mask = pad_sequence_to_length(char_identity_mask, length=max_len, value=-1)

            if self.use_charbank in ['global-ce', 'global-cae']:
                LEN = 10
                exampler_attn_mask = pad_sequence_to_length([torch.ones(i.shape[0]) for i in exampler_feature_list], 
                                                            length=LEN, value=0)
                exampler_feature_list = pad_sequence_to_length(exampler_feature_list, 
                                                               length=LEN, value=0)
                return_dict['exampler_attn_mask'] = exampler_attn_mask
                return_dict['exampler_feature'] = exampler_feature_list
        return_dict['char_tokens'] = char_tokens
        return_dict['char_text'] = char_list_with_prompt
        return_dict['char_identity'] = char_identity_mask
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['video'] = pad_sequence_by_last([sample['video'] for sample in batch])
        out_batch['padding_mask'] = pad_sequence([sample['padding_mask'] for sample in batch], batch_first=True, padding_value=1.0)
        out_batch['text'] = [sample['text'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['vid'] = [sample['vid'] for sample in batch]
        out_batch['token'] = [sample['token'] for sample in batch]
        out_batch['binary_tgt'] = default_collate([sample['binary_tgt'] for sample in batch])
        if 'char_tokens' in batch[0]:
            out_batch['char_tokens'] = pad_sequence_by_dim([sample['char_tokens'] for sample in batch], dim=1, value=EOS_TOKEN_ID)
            out_batch['char_text'] = [sample['char_text'] for sample in batch]
            out_batch['char_identity'] = pad_sequence_by_dim([sample['char_identity'] for sample in batch], dim=1, value=-1)
        if 'exampler_feature' in batch[0]:
            out_batch['exampler_feature'] = default_collate([sample['exampler_feature'] for sample in batch])
            out_batch['exampler_attn_mask'] = default_collate([sample['exampler_attn_mask'] for sample in batch])
        return out_batch 

    def __len__(self):
        return self.dataset_length
    




if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    D = LSMDC_NameLoader(use_charbank='global-ce', force_resample=False, tokenizer=tokenizer, exclude_movienet=True)
    import ipdb; ipdb.set_trace()

    D_mad = MAD_NameLoader(use_charbank='global-ce', force_resample=False, tokenizer=tokenizer)
    D = MovieNet_NameLoader(use_charbank='global-ce', force_resample=False, tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(D, batch_size=1, num_workers=0,
        collate_fn=D.collate_fn)

    import ipdb; ipdb.set_trace()

    for output in tqdm(loader, total=len(loader)):
        print(output['text'])
