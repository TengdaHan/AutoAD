import os
from openai import OpenAI
import argparse
import json
import ast
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Process, Queue


# Execute function with a timeout
# OpenAI API call sometimes freezes
def my_function(result_queue, fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    result_queue.put(result)

def run_with_timeout(fn, *args,):
    result_queue = Queue()
    while True:
        p = Process(target=my_function, args=(result_queue, fn, *args,))
        p.start()
        p.join(timeout=3)  # Set the timeout value (in seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            print("Function timed out, redo ...")
        else:
            result = result_queue.get()
            break
    return result


def eval_each(text_gt, text_pred, client):
    """ Compute the LLM score for one pair """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the quality of generative outputs for movie audio descriptions. "
                    "Your task is to compare the predicted audio descriptions with the correct audio descriptions and determine its level of match, considering mainly the visual elements like actions, objects and interactions. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Check if the predicted audio description covers the main visual events from the movie, especially focusing on the verbs and nouns.\n"
                    "- Evaluate whether the predicted audio description includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                    "- Consider synonyms or paraphrases as valid matches. Consider pronouns like 'he' or 'she' as valid matches with character names. Consider different character names as valid matches. \n"
                    "- Provide a single evaluation score that reflects the level of match of the prediction, considering the visual elements like actions, objects and interactions."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following movie audio description pair:\n\n"
                    f"Correct Audio Description: {text_gt}\n"
                    f"Predicted Audio Description: {text_pred}\n\n"
                    "Provide your evaluation only as a matching score where the matching score is an integer value between 0 and 5, with 5 indicating the highest level of match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the matching score in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'score': }."
            }
        ]
    )
    # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content
    response_dict = ast.literal_eval(response_message)
    del completion
    return [response_dict]


def main(args):
    """ Main function to control the flow of the program. """
    client = OpenAI()
    eval_fn = partial(eval_each, client=client)

    pred_df = pd.read_csv(args.path)
    
    all_output = []
    gt_pred_pair = [(x, y) for x,y in zip(pred_df['text_gt'], pred_df['text_gen'])]

    # save output regularly, in case api call breaks
    chunk_size = 200
    num_chunk = (len(gt_pred_pair) // chunk_size) + 1

    for chunk_idx in tqdm(range(num_chunk)):
        if chunk_idx != 0:
            with open(f'tmp/log_{chunk_idx:05d}.json', 'w') as fobj:
                fobj.write(all_output)

        gt_pred_pair_current = gt_pred_pair[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]
        for (gt, pred) in tqdm(gt_pred_pair_current, total=len(gt_pred_pair_current)):
            result = run_with_timeout(eval_fn, gt, pred)
            all_output.append(result)
    
    all_score = []
    for i in all_output:
        try:
            all_score.append(i[0]['score'])
        except:
            print(i, 'does not follow the format, skip.')
            continue
        
    print(np.mean(all_score))
    with open(f'tmp/log_final.json', 'w') as fobj:
        fobj.write(all_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="inference output in csv file. Require 'text_gt' and 'text_gen' columns.")
    parser.add_argument("--api_key", required=False, default=None, help="OpenAI API key.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.api_key is not None:
        os.environ['OPENAI_API_KEY'] = args.api_key
    # otherwise set OPENAI_API_KEY by running: export OPENAI_API_KEY='your-api-key-here'
    main(args)
