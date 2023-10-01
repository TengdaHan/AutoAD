"""Find in-movie exemplars for character's profile pictures in MovieNet.
The source is from AudioVault charbank download.
This script works on the intersection:
(set(MovieNet) + set(AudioVault)) - set(MAD)
and process them.
Output: the {imdbid}.charbank.pth.tar files in /scratch/shared/beegfs/htd/DATA/MovieNet/charbank_cos_top10
"""

import pandas as pd
import json
import numpy as np
import torch
import tqdm
import os
import sys
import matplotlib.pyplot as plt
from glob import glob
import re


def draw_sim_curve(curves, char_list):
    assert len(curves) == len(char_list)
    if isinstance(curves[0], torch.Tensor):
        curves = torch.stack(curves, 0)
    else:
        curves = np.stack(curves, 0)
    fig, ax = plt.subplots(figsize=(18,4))
    for i in range(len(char_list)):
        ax.plot(curves[i], label=char_list[i])
    ax.legend()
    return fig, ax


def get_imdb_to_process():
    """a set operation: (set(MovieNet) + set(AudioVault)) - set(MAD)"""
    # audiovault
    av_id = pd.read_csv('/scratch/shared/beegfs/maxbain/datasets/audiovault/audiovault_imdbids.csv')
    av_id_matched = av_id[av_id['exact_match']]['imdbid'].tolist()
    # movienet
    movienet_anno = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/movienet/movienet_face_anno.csv')
    mvn_id = movienet_anno['movie_id'].unique().tolist()
    # mad
    mad_imdb = pd.read_csv('/scratch/shared/beegfs/htd/DATA/MAD/mad_imdb_info.csv')
    mad_id = mad_imdb['imdb'].unique().tolist()
    
    imdb_list = list((set(mvn_id).intersection(set(av_id_matched))) - set(mad_id))
    print(f"{len(imdb_list)=}")
    return sorted(imdb_list), "v1"


def get_imdb_to_process_outside_av():
    """a set operation: not in audiovault, but in the downloaded address"""
    # audiovault
    av_id = pd.read_csv('/scratch/shared/beegfs/maxbain/datasets/audiovault/audiovault_imdbids.csv')
    av_id_matched = av_id[av_id['exact_match']]['imdbid'].tolist()
    # movienet
    movienet_anno = pd.read_csv('/work/htd/Desktop_tmp/AutoMad/movienet/movienet_face_anno.csv')
    mvn_id = movienet_anno['movie_id'].unique().tolist()
    # mad
    mad_imdb = pd.read_csv('/scratch/shared/beegfs/htd/DATA/MAD/mad_imdb_info.csv')
    mad_id = mad_imdb['imdb'].unique().tolist()
    # recent download
    download_list = glob("/scratch/shared/beegfs/maxbain/datasets/audiovault/cast_mainpage_selenium/*.json")
    download_id = [os.path.basename(i).split('.')[0] for i in download_list]

    imdb_list = list((set(mvn_id) - set(av_id_matched) - set(mad_id)).intersection(set(download_id)))
    print(f"{len(imdb_list)=}")
    return sorted(imdb_list), "v2"


MAD_dir = "/scratch/shared/beegfs/maxbain/datasets/MAD_Language_Grounding_Movie_Audio_Descriptions"
audiovault_dir = "/scratch/shared/beegfs/maxbain/datasets/audiovault"
movienet_charbank_json_dir = "/scratch/shared/beegfs/maxbain/datasets/audiovault/cast_mainpage_selenium"
char_bank_exp_fn = "../data_post_proc/MAD_charbank_2023mar.json"
TOP_K_CHARS = 10


# Load raw info from audiovault
char_map = pd.read_csv(os.path.join(audiovault_dir, "audiovault_actors.csv"))
# char_map = char_map.groupby("imdbid").head(TOP_K_CHARS)

face_data = torch.load('audiovault_face_ViT-L-14.pth.tar')
profile_list = face_data['filenames']
profile_list = [i.split('.')[0] for i in profile_list]
profile_features = face_data['clip_embedding']
assert len(profile_list) == profile_features.shape[0]


# two batches of download
# get imdb_id
for get_imdb_fn in [get_imdb_to_process, get_imdb_to_process_outside_av]:
    # iterate:
    #   movienet_imdb_list, version = get_imdb_to_process()
    #   movienet_imdb_list, version = get_imdb_to_process_outside_av()

    movienet_imdb_list, version = get_imdb_fn()
    # check feature exists
    feature_root = "/scratch/shared/beegfs/htd/DATA/MovieNet/keyframe_feat/openai-clip-vit-l-14"
    print(f"{len(movienet_imdb_list)=}")
    movienet_imdb_list = [i for i in movienet_imdb_list if os.path.exists(os.path.join(feature_root, f"{i}.npy"))]
    print(f"{len(movienet_imdb_list)=}")

    failed_char = 0
    char_dict = {}
    save_root = "/scratch/shared/beegfs/htd/MovieNet/charbank_cos_top10"
    os.makedirs(save_root, exist_ok=True)

    # what do we get: char info array
    for imdbid in tqdm.tqdm(movienet_imdb_list):
        if version == 'v1':
            pkl_path = os.path.join(audiovault_dir, f'imdb/{imdbid}.pkl')
            if not os.path.exists(pkl_path):
                print(f"{pkl_path} does not exist")
                continue
            imdb_dump = np.load(pkl_path, allow_pickle=True)
            assert imdbid.replace('tt','') == imdb_dump['imdbID']
            cast_info = imdb_dump['cast'][:TOP_K_CHARS]
        elif version == 'v2':
            json_path = os.path.join(movienet_charbank_json_dir, f'{imdbid}.json')
            cast_info = json.load(open(json_path))
            cast_info = cast_info[:TOP_K_CHARS]

        # get CLIP feature: default L14
        v_feature = torch.from_numpy(np.load(f"{feature_root}/{imdbid}.npy")).float().cuda()
        v_feature_normed = v_feature / v_feature.norm(dim=-1, keepdim=True)
        char_dicts = []

        for cdx, cinfo in enumerate(cast_info):
            if isinstance(cinfo, dict):
                curr_role = cinfo['role']
                role = curr_role
                role = re.sub(r'\(.*?\)', '', role)
                person_id = cinfo['id'].replace('nm', '')
                name = cinfo['name']
                long_name = name
            else:
                curr_role = cinfo._get_currentRole()
                # ignore multi-role actor cases for now
                if isinstance(curr_role, list):
                    curr_role = curr_role[0]
                role = curr_role['name']
                role = re.sub(r'\(.*?\)', '', role)
                person_id = cinfo.personID
                name = cinfo['name']
                long_name = cinfo['long imdb name']

            char_info_array = {
                "id": int(person_id),
                "name": str(name),
                "long imdb name": str(long_name),
                "role": str(role),
            }

            if 'nm'+str(person_id) in profile_list:
                ftr_idx = profile_list.index('nm'+str(person_id))
                ftr_profile = profile_features[ftr_idx].float()[None,:].cuda()
                
                ftr_profile_normed = ftr_profile / ftr_profile.norm(dim=-1, keepdim=True)
                cos_curve = v_feature_normed @ ftr_profile_normed[0,]
                # get top5 faces in the movie then average
                # adaptive top5 for diversity: with a gap of 10 shots (3 KeyFrames per shot)
                HALF_WINDOW = 10 * 3
                cos_curve_copy = cos_curve.clone()
                topkidx_list = []
                topkval_list = []
                for _ in range(5):
                    max_val, max_idx = torch.max(cos_curve_copy, dim=-1)
                    topkidx_list.append(max_idx)
                    topkval_list.append(max_val)
                    cos_curve_copy[max(0, max_idx-HALF_WINDOW): min(cos_curve_copy.shape[0], max_idx+HALF_WINDOW)] = -1
                topkidx = torch.stack(topkidx_list, 0)
                # _, topkidx = torch.topk(cos_curve, k=5, dim=-1)

                avg_profile = v_feature_normed[topkidx].mean(0, keepdim=True)
                cos_curve_self = v_feature_normed @ avg_profile[0,]
            else:
                print(f'{imdbid} {char_info_array["role"]} cannot be found in profile_list')
                failed_char += 1
                cos_curve = torch.ones(v_feature_normed.shape[0], device='cuda') * -float('inf')
                cos_curve_self = torch.ones(v_feature_normed.shape[0], device='cuda') * -1
                topkidx = torch.ones(5, device='cuda') * -1

            char_info_array.update({
                "cos_curve": cos_curve.cpu(),
                "cos_curve_self_top5": cos_curve_self.cpu(),
                'top5_idx': topkidx.cpu()}
            )
            char_dicts.append(char_info_array)
        
        try:
            stack_cos = torch.stack([i['cos_curve_self_top5'] for i in char_dicts], 0)
        except:
            print("warning: stack COS failed")
            stack_cos = torch.zeros(TOP_K_CHARS, v_feature_normed.shape[0])

        per_movie_info = {'roles': [i['role'] for i in char_dicts],
            'names': [i['name'] for i in char_dicts],
            'cos': stack_cos,
            'top5_idx': [i['top5_idx'] for i in char_dicts],
            }
        
        per_movie_info_simple = [{'id': 'nm'+f"{item['id']:07d}", 'name': item['name'], 'role': item['role']} for item in char_dicts]
        char_dict[imdbid] = per_movie_info_simple
        save_name = f'{imdbid}.charbank.pth.tar'
        torch.save(per_movie_info, os.path.join(save_root, save_name))

    print(f"{failed_char=}")
    print(f"total char should be {len(movienet_imdb_list) * 10}")
    print(f"failed ratio = {failed_char/(len(movienet_imdb_list)*10)}")

    if version == 'v1':
        with open("MovieNet_charbank_300.json", 'w') as fobj:
            json.dump(char_dict, fobj)
    elif version == 'v2':
        with open("MovieNet_charbank_patch_148.json", 'w') as fobj:
            json.dump(char_dict, fobj)

print(f'finished, saved to {save_root}')
sys.exit(0)
