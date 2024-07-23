""" CRITIC (Co-Referencing In Text for Identifying Characters) metric.

This script evaluates the character identification between list of predicted sentences and reference sentences.
It has two steps:
    Step1: build synonym set for each identity in each sentence (based on "fastcoref"), for both GT and prediction
    Step2: compute identity IoU for each sentence, then aggregate

This metric is based on the co-reference package "F-COREF": https://pypi.org/project/fastcoref/     
"""

import pandas as pd
from fastcoref import FCoref
import numpy as np
from tqdm import tqdm
import json
import argparse

coref_model = FCoref(device='cuda:0', enable_progress_bar=False)


def build_synonym(coref_data, source_idx, role_names, drop_pronouns=True):
    """ Function to extract clusters containing any of the character names. """
    res = []
    text = coref_data.text
    total_rows = np.max(source_idx) + 1
    synonym_rows = {idx: [] for idx in np.arange(total_rows)}
    synonym_rows_cid = {idx: [] for idx in np.arange(total_rows)}
    synonym_rows_origin = {idx: [] for idx in np.arange(total_rows)}

    for _, cluster in enumerate(coref_data.get_clusters(as_strings=False)):
        cluster_name = None
        # some cluster is char name; some is not (e.g. a letter, it, the letter)
        cluster_str_origin = [text[x[0]:x[1]] for x in cluster]
        cluster_str = [text[x[0]:x[1]] for x in cluster]
        match_role_set = set(cluster_str).intersection(set(role_names))
        IS_CHAR = len(match_role_set) > 0
        if IS_CHAR:
            if not len(match_role_set) == 1:
                # if a coref result match multiple characters, it is not a good coref; discard it
                print(f'Warning: found a bad coref {set(cluster_str)} vs. {role_names}, continue')
                continue
            cluster_name = list(match_role_set)[0]
            # assign the synonym set back to each data row
            cluster_source_idx = [source_idx[x[0]:x[1]] for x in cluster]
            if len(cluster_name.split()) > 1:  # has "first last", or "title last", or "title first last"
                cluster_str.extend([cluster_name.split()[0], cluster_name.split()[-1], 
                                    cluster_name.split()[0].lower(), cluster_name.split()[-1].lower(),
                                    cluster_name.split()[0].upper(), cluster_name.split()[-1].upper()])
            synonym_set = list(set(cluster_str))
            if drop_pronouns:
                synonym_set = [i for i in synonym_set if i.lower() not in ['she', 'he', 'her', 'his', 'they']]

            for item, text in zip(cluster_source_idx, cluster_str_origin):
                if not np.mean(item) == np.max(item):
                    continue
                if item[0] != -1:
                    if cluster_name not in synonym_rows_cid[item[0]]:  # dedup
                        synonym_rows[item[0]].append(synonym_set)
                        synonym_rows_cid[item[0]].append(cluster_name)
                        synonym_rows_origin[item[0]].append(text)

    res = {k:list(v) for k,v in synonym_rows.items()}
    return res, synonym_rows_origin, synonym_rows_cid


def get_iou(list1, list2):
    """Get IoU of two lists of strings"""
    intersection = set(list1).intersection(set(list2))
    union = set(list1).union(set(list2))
    if len(union) == 0:
        return 0
    else:
        return len(intersection)/len(union)


def coref_metric(df, character_list):
    roles_str = ""
    if len(character_list) > 1:
        roles_str = ', '.join(character_list[:-1]) + ' and '
    if len(character_list) > 0:
        roles_str += character_list[-1] + '.'

    ### prepare GT and pred
    # dataframe should have keys 'text_gt' and 'text_gen'

    # Keep the row index of each character string (like "a" "1" ".". not movie characters),
    # because FCoref returns the string indexes of each identity (e.g. "Jack" with position [110, 114]),
    # we want to know which sentence each Coref output comes from.
    # e.g. ["Jack smiles", "He stands up"] 
    #  --> [[0,0,0,0,0, 0,0,0,0,0, 0], [1,1,1,1,1, 1,1,1,1,1, 1,1]]

    # FCoref gives character string index starting from 1, rather than 0.
    # Therefore we prepend "-1" as a placeholder.
    # e.g. ["Jack smiles", "He stands up"] 
    #  --> [[-1, 0,0,0,0,0, 0,0,0,0,0, 0], [-1, 1,1,1,1,1, 1,1,1,1,1, 1,1]]
    gt_text = ' '.join(df['text_gt'].tolist())
    gt_source_idx_list = [[i]*len(x) for i,x in enumerate(df['text_gt'].tolist())]
    gt_source_idx = []
    for i, item in enumerate(gt_source_idx_list):
        if i != 0:
            item = [-1] + item
        gt_source_idx.extend(item)
    assert len(gt_source_idx) == len(gt_text)  
    

    pred_text = ' '.join(df['text_gen'].tolist())
    pred_source_idx_list = [[i]*len(x) for i,x in enumerate(df['text_gen'].tolist())]
    pred_source_idx = []
    for i, item in enumerate(pred_source_idx_list):
        if i != 0:
            item = [-1] + item
        pred_source_idx.extend(item)
    assert len(pred_source_idx) == len(pred_text)

    # we prepend the cast list for Coref model, 
    # we also prepend multiple -1s to the index list as placeholders
    # e.g. ["Jack and Rose", "Jack smiles", "He stands up"]
    #  --> [[-1]*13, 
    #       [-1, 0,0,0,0,0, 0,0,0,0,0, 0], 
    #       [-1, 1,1,1,1,1, 1,1,1,1,1, 1,1]]
    gt_source_idx = [-1] * (len(roles_str)+1) + gt_source_idx
    pred_source_idx = [-1] * (len(roles_str)+1) + pred_source_idx


    ### Compute coref, get identity clusters
    coref_gts = coref_model.predict(
       texts=[f"{roles_str} {gt_text}"]
    )[0]
    assert len(gt_source_idx) == len(coref_gts.text)

    coref_preds = coref_model.predict(
       texts=[f"{roles_str} {pred_text}"]
    )[0]
    assert len(pred_source_idx) == len(coref_preds.text)

    ### Compute synonym set for each sentence
    synonym_rows_gt, synonym_origin_gt, synonym_cid_gt = build_synonym(coref_gts, gt_source_idx, character_list)
    assert len(df) == len(synonym_rows_gt)
    synonym_rows_pred, synonym_origin_pred, synonym_cid_pred = build_synonym(coref_preds, pred_source_idx, character_list)
    assert len(df) == len(synonym_rows_pred)

    # Rewrite text with fullnames to reduce ambiguilty
    gt_sentence_list = df['text_gt'].tolist()
    assert len(gt_sentence_list) == len(synonym_origin_gt)
    fullname_gt_sentence_list = []
    for s_idx, ps in enumerate(gt_sentence_list):
        origin_words = synonym_origin_gt[s_idx]
        cids = synonym_cid_gt[s_idx]
        fullname_sentence = ps
        for ow, cid in zip(origin_words, cids):
            fullname_sentence = fullname_sentence.replace(ow, cid)
        fullname_gt_sentence_list.append(fullname_sentence)
    df['fullname_gt'] = fullname_gt_sentence_list

    pred_sentence_list = df['text_gen'].tolist()
    assert len(pred_sentence_list) == len(synonym_origin_pred)
    fullname_pred_sentence_list = []
    for s_idx, ps in enumerate(pred_sentence_list):
        origin_words = synonym_origin_pred[s_idx]
        cids = synonym_cid_pred[s_idx]
        fullname_sentence = ps
        for ow, cid in zip(origin_words, cids):
            fullname_sentence = fullname_sentence.replace(ow, cid)
        fullname_pred_sentence_list.append(fullname_sentence)
    df['fullname_pred'] = fullname_pred_sentence_list

    ### Aggregate results for each sentence
    iou_list = []
    iou_list_for_df = []

    for row_idx in tqdm(range(len(df))):
        synonym_set = synonym_rows_gt[row_idx]
        num_set = len(synonym_set)
        if num_set == 0:
            continue
        iou_list.append(get_iou(synonym_cid_gt[row_idx], synonym_cid_pred[row_idx]))
        iou_list_for_df.append(get_iou(synonym_cid_gt[row_idx], synonym_cid_pred[row_idx]))

    return iou_list, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                        default='your_path/inference.csv',
                        help="inference output in csv file. Require 'vid', 'text_gt' and 'text_gen' columns.")
    args = parser.parse_args()

    # Helper functions specific to CMD
    with open("cast_list_for_eval.json", 'r') as fobj:
        imdbid_to_cast = json.load(fobj)

    with open("cmd_fn_to_imdb.json", 'r') as fobj:
        cmd_fn_to_imdb = json.load(fobj)

    pred_df = pd.read_csv(args.path)

    # vid refers to CMD filenames (each movie has 10-20 clips), 
    # here we found the IMDB ID of their source movie.
    pred_df['imdbid'] = pred_df.apply(lambda x: cmd_fn_to_imdb[x['vid']], axis=1)
    pred_df['text_gen'] = pred_df['text_gen'].astype(str)

    # we assume after grouping by "imdbid", each group is temporally sorted
    # otherwise Coref does not make sense
    val_df_dict = dict(tuple(pred_df.groupby('imdbid')))

    total_iou = []

    for imdbid, df in tqdm(val_df_dict.items(), total=len(val_df_dict)):
        char_list = imdbid_to_cast[imdbid]
        iou_list, _ = coref_metric(df, char_list)
        total_iou.extend(iou_list)

    print(f"avg IoU on {len(total_iou)} predictions with identities", np.mean(total_iou))