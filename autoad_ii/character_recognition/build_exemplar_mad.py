"""Find in-movie exemplars for character's profile pictures in MAD,
by taking top-K nearest neighbour"""

import pandas as pd
import json
import numpy as np
import torch
import tqdm
import os
import matplotlib.pyplot as plt
import glob


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


MAD_dir = "/scratch/shared/beegfs/maxbain/datasets/MAD_Language_Grounding_Movie_Audio_Descriptions"
audiovault_dir = "/scratch/shared/beegfs/maxbain/datasets/audiovault"
char_bank_exp_fn = "MAD_charbank_2023mar.json"
TOP_K_CHARS = 10

# load charbank json
with open(os.path.join(char_bank_exp_fn)) as fobj:
    charbank_dict = json.load(fobj)

# available at: wget http://www.robots.ox.ac.uk/~htd/autoad/MAD_id2imdb.json
with open(os.path.join(MAD_dir, "MAD_id2imdb.json"), "r") as fobj:
    imdb_map = json.load(fobj)

# available at: wget http://www.robots.ox.ac.uk/~htd/autoad/audiovault_actors.csv
char_map = pd.read_csv(os.path.join(audiovault_dir, "audiovault_actors.csv"))

# using CLIP-L-14
# available at: wget http://www.robots.ox.ac.uk/~htd/autoad/audiovault_face_ViT-L-14.pth.tar
face_data = torch.load('audiovault_face_ViT-L-14.pth.tar')
profile_list = face_data['filenames']
profile_list = [i.split('.')[0] for i in profile_list]
profile_features = face_data['clip_embedding']
assert len(profile_list) == profile_features.shape[0]

char_dict = {}
TOP_K = 5
failed_avid = 0
failed_char = 0

print(f"computing charbank for top{TOP_K_CHARS} characters for top{TOP_K} features")

save_root = '/scratch/shared/beegfs/htd/MAD/charbank_cos_top10_cal_2023jul'
os.makedirs(save_root, exist_ok=True)
feature_root = '/scratch/shared/beegfs/htd/DATA/MAD/CLIP_L14_frames_features_5fps'
avid_list = sorted(glob.glob(os.path.join(feature_root, '*.npy')))
avid_list = [os.path.basename(i).replace('.npy', '') for i in avid_list]
print(f'Get {len(avid_list)} movie features from {feature_root}')


for avid in tqdm.tqdm(avid_list):
    imdbid = imdb_map[avid]
    cdf = char_map[char_map['imdbid']==imdbid]
    cdf = cdf.reset_index()
    cdf['cast_index'] = cdf.index

    if avid not in charbank_dict:
        print(f'{avid} is not downloaded in charbank_dict')
        failed_avid += 1
        continue

    imdb_dump = charbank_dict[avid]
    cast_info = imdb_dump[:TOP_K_CHARS]

    movie_feature = torch.from_numpy(np.load(os.path.join(feature_root, f'{avid}.npy'))).float().cuda()
    movie_feature_normed = movie_feature / movie_feature.norm(dim=-1, keepdim=True)

    curves = []
    curves_self = []
    char_dicts = []
    all_cos_curves = []

    for char_info in cast_info:
        char_id_int = int(char_info['id'].replace('nm', ''))
        crows = cdf[cdf['id'] == char_id_int]
        assert len(crows) == 1
        crow = crows.iloc[0]
        assert int(char_info['id'].replace('nm','')) == crow['id']                
        curr_role = char_info['role']

        # ignore multi-role actor cases for now
        if isinstance(curr_role, list):
            curr_role = curr_role[0]
        crow_id_str = f"{crow['id']:07d}"

        if char_info['id'] in profile_list:
            profile_idx = profile_list.index(char_info['id'])
            profile_ftr = profile_features[profile_idx].float()[None,:].cuda()

            # cos_curve = torch.nn.functional.cosine_similarity(curr_MAD_ftrs, ftr_profile)
            ftr_profile_normed = profile_ftr / profile_ftr.norm(dim=-1, keepdim=True)
            cos_curve = movie_feature_normed @ ftr_profile_normed[0,]

            # get top5 features in the movie then average
            # adaptive top5 for diversity: with a gap of 10second (5FPS)
            HALF_WINDOW = 10 * 5
            cos_curve_copy = cos_curve.clone()
            topkidx_list = []
            topkval_list = []
            for _ in range(5):
                max_val, max_idx = torch.max(cos_curve_copy, dim=-1)
                topkidx_list.append(max_idx)
                topkval_list.append(max_val)
                cos_curve_copy[max(0, max_idx-HALF_WINDOW): min(cos_curve_copy.shape[0], max_idx+HALF_WINDOW)] = -1
            topkidx = torch.stack(topkidx_list, 0)

            # average top5 features as the exemplar
            avg_profile = movie_feature_normed[topkidx].mean(0, keepdim=True)
            cos_curve_self = movie_feature_normed @ avg_profile[0,]
        else:
            print(f'{avid} {curr_role} cannot be found in profile_list')
            failed_char += 1
            cos_curve = torch.ones(movie_feature_normed.shape[0], device='cuda') * -float('inf')
            cos_curve_self = torch.ones(movie_feature_normed.shape[0], device='cuda') * -1
            topkidx = torch.ones(5, device='cuda') * -1

        char_info_array = {
            "id": int(char_info['id'].replace('nm','')),
            "name": str(char_info['name']),
            "long imdb name": str(char_info['name']),
            "role": str(char_info['role']),
            "cos_curve": cos_curve.cpu(),
            "cos_curve_self_top5": cos_curve_self.cpu(),
            'top5_idx': topkidx.cpu(),
        }
        curves.append(char_info_array['cos_curve'])
        curves_self.append(char_info_array['cos_curve_self_top5'])
        char_dicts.append(char_info_array)

    stack_cos = torch.stack([i['cos_curve_self_top5'] for i in char_dicts], 0)
    per_movie_info = {
        'roles': [i['role'] for i in char_dicts],
        'names': [i['name'] for i in char_dicts],
        'cos': stack_cos,
        'top5_idx': [i['top5_idx'] for i in char_dicts],
        }
    char_dict[avid] = per_movie_info

    save_name = f'{avid}.charbank.pth.tar'
    torch.save(per_movie_info, os.path.join(save_root, save_name))

print('finished')
print(f'failed avid: {failed_avid}')
print(f'failed char: {failed_char}')
print(f'saved to {save_root}')

