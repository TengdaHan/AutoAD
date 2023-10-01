"""Modified from https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing#scrollTo=OArDkm_24w4L """

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
# from transformers.generation_utils import top_k_top_p_filtering

@torch.no_grad()
def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None, attention_mask=None,
                  entry_length=67, temperature=1., stop_token: str = '.', past_key_values=None, media=None,
                  repetition_penalty=1.2,
                  no_repeat_ngram_size=3,
                  history_tokens=None,):

    model.eval()
    # stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_index = 50256
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    if embed is not None:
        generated = embed
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompt)).long()
            tokens = tokens.unsqueeze(0).to(device)
            generated = model.gpt.transformer.wte(tokens)
    
    for i in range(entry_length):
        if media is not None:
            outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values, media=media)
        else:
            outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values)

        if past_key_values is not None:
            past_key_values = outputs.past_key_values
        
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = F.log_softmax(logits, -1)
        logits = enforce_repetition_penalty(logits, tokens, repetition_penalty=repetition_penalty)

        # # entire generated paragraph as ngram repetition penalty
        if (history_tokens is None) or (history_tokens.numel() == 0):
            h_tokens = tokens
        elif tokens is None:
            h_tokens = history_tokens.repeat(beam_size, 1)
        else:
            h_tokens = torch.cat((history_tokens.repeat(beam_size, 1), tokens), dim=1)
        cul_len = 0 if h_tokens is None else h_tokens.shape[1]
        num_hypo = 1 if h_tokens is None else h_tokens.shape[0]
        banned_batch_tokens = calc_banned_ngram_tokens(
            h_tokens, num_hypo, no_repeat_ngram_size, cul_len
        )
        if len(banned_batch_tokens) > logits.shape[0]:
            banned_batch_tokens = [banned_batch_tokens[0]]
        for i, banned_tokens in enumerate(banned_batch_tokens):
            logits[i, banned_tokens] = -float("inf")
        
        # beam search
        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
            next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')  # next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        if past_key_values is not None:
            generated = next_token_embed
            if i == 0:
                past_key_values = tuple([(tp[0].repeat(beam_size,1,1,1), tp[1].repeat(beam_size,1,1,1)) for tp in past_key_values])
        else:
            generated = torch.cat((generated, next_token_embed), dim=1)
        
        if attention_mask is not None:
            if i == 0:
                attention_mask = attention_mask.repeat(beam_size, 1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:,-2:-1])), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


@torch.no_grad()
def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty=1.0):
    if (prev_output_tokens is None) or (prev_output_tokens.shape == 0) or (repetition_penalty == 1.0):
        return lprobs
    for i in range(lprobs.shape[0]):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0: # log(prob)
                lprobs[i, previous_token] *= repetition_penalty
            else:  # prob
                lprobs[i, previous_token] /= repetition_penalty
    return lprobs


@torch.no_grad()
def calc_banned_ngram_tokens(prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])
    
    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


@torch.no_grad()
def generate_greedy(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        attention_mask=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
        verbose=False,
        past_key_values=None,
        media=None,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        history_tokens=None,
):
    model.eval()
    generated_num = 0
    generated_list = []
    # stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_index = 50256
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    for entry_idx in trange(entry_count, disable=not verbose):
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt)).long()
                tokens = tokens.unsqueeze(0).to(device)

            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            if media is not None:
                outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values, media=media)
            else:
                outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values)
            if past_key_values is not None:
                past_key_values = outputs.past_key_values
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = enforce_repetition_penalty(F.log_softmax(logits, dim=-1), tokens, repetition_penalty=repetition_penalty)

            # entire generated paragraph as ngram repetition penalty
            if (history_tokens is None) or (history_tokens.numel() == 0):
                h_tokens = tokens
            elif tokens is None:
                h_tokens = history_tokens
            else:
                h_tokens = torch.cat((history_tokens, tokens), dim=1)
            cul_len = 0 if h_tokens is None else h_tokens.shape[1]
            banned_batch_tokens = calc_banned_ngram_tokens(
                h_tokens, 1, no_repeat_ngram_size, cul_len
            )

            for i, banned_tokens in enumerate(banned_batch_tokens):
                logits[i, banned_tokens] = -float("inf")

            logits = F.softmax(logits, dim=-1)

            # TOP-P filtering
            # sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # sorted_indices_to_remove = cumulative_probs > top_p
            # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            #                                     ..., :-1
            #                                     ].clone()
            # sorted_indices_to_remove[..., 0] = 0

            # indices_to_remove = sorted_indices[sorted_indices_to_remove]
            # logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            
            if past_key_values is not None:
                generated = next_token_embed
            else:
                generated = torch.cat((generated, next_token_embed), dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:,-2:-1])), dim=1)
            if stop_token_index == next_token.item():
                break

        try:
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list, skip_special_tokens=True)
        except:
            output_text = '.'
        
        generated_list.append(output_text)

    return generated_list[0]


@torch.no_grad()
def generate_top_k_top_p(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        attention_mask=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        top_k=3,
        temperature=1.,
        stop_token: str = '.',
        verbose=False,
        past_key_values=None,
        media=None,
):
    """modified from https://github.com/JasonBenn/duet/blob/master/generate.py"""

    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    for entry_idx in trange(entry_count, disable=not verbose):
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt)).long()
                tokens = tokens.unsqueeze(0).to(device)

            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            if media is not None:
                outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values, media=media)
            else:
                outputs = model.gpt(inputs_embeds=generated, attention_mask=attention_mask, past_key_values=past_key_values)
            if past_key_values is not None:
                past_key_values = outputs.past_key_values
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            
            if past_key_values is not None:
                generated = next_token_embed
            else:
                generated = torch.cat((generated, next_token_embed), dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:,-2:-1])), dim=1)
            if stop_token_index == next_token.item():
                break

        try:
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
        except:
            output_text = '.'
        
        generated_list.append(output_text)

    return generated_list[0]


@torch.no_grad()
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # indices_to_remove = sorted_indices[sorted_indices_to_remove][None,:]
        indices_to_remove = torch.zeros_like(logits, dtype=torch.long).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove.long()).bool()
        logits[indices_to_remove] = filter_value
    return logits

