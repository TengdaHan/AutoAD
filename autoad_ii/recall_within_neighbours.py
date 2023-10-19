import torch
from bert_score import BERTScorer  # from https://github.com/Tiiiger/bert_score
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def recall_within_neighbours(sentences_gt, sentences_gen, topk=(1,5), N=16):
    """compute R@k/N as described in AutoAD-II (https://www.robots.ox.ac.uk/~vgg/publications/2023/Han23a/han23a.pdf)
    This metric compares a (long) list of sentences with another list of sentences.
    It uses BERTScore (https://github.com/Tiiiger/bert_score) to compute sentence-sentence similarity,
    but uses the relative BERTScore values to get a recall, for robustness.
    """
    # get sentence-sentence BertScore
    ss_score = []
    for sent in sentences_gen:
        ss_score.append(bert_scorer.score(sentences_gt, [sent] * len(sentences_gt))[-1])
    ss_score = torch.stack(ss_score, dim=0)

    window = N
    topk_output = []
    for i in range(0, ss_score.shape[0]-window+1, window//2):
        topk_output.append(calc_topk_accuracy(ss_score[i:i+window,i:i+window], torch.arange(window).to(ss_score.device), topk=topk))
    
    topk_avg = torch.stack(topk_output, 0).mean(0).tolist()
    for k, res in zip(topk, topk_avg):
        print(f"Recall@{k}/{N}: {res:.3f}")
    return topk_avg


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return torch.stack(res)


if __name__ == '__main__':
    # Example
    # in practice, we put all the ADs of a movie in sentences_gt and sentences_gen
    sentences_gt = [
        "Inside the flat, Tony switches on the light.",
        "Mark and Margo and Inspector Hubbard stand facing him.",
        "He quickly opens the door.",
        "A detective stands outside.",
        "Tony slowly closes the door again.",
        "Distressed, Margo turns away.",
        "Tony shrugs, looks at the drinks on the sideboard, then walks calmly to the desk and picks up a bottle of whisky.",
        "He puts the key on the desk.",
        "Taking a glass from the sideboard, he pours himself a large drink.",
        ]
    sentences_gen = [
        "Tony stands.",
        "They stand facing him.",
        "He stand.",
        "A man stands.",
        "Tony closes the door.",
        "Margo turns away.",
        "Tony walks around.",
        "He walks towards the desk.",
        "He holds a book.",
    ]
    result = recall_within_neighbours(sentences_gt, sentences_gen, topk=(1,3), N=4)
    # Should get
    # Recall@1/4: 0.667
    # Recall@3/4: 0.917

