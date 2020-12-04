import argparse
import json
import os

from feqa import FEQA


def main(args):
    documents = []
    summaries = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            instance = json.loads(line)
            documents.append(instance['document'])
            summaries.append(instance['summary'])

    scorer = FEQA(use_gpu=True)
    scores = scorer.compute_score(documents, summaries, aggregate=False)

    dirname = os.path.dirname(args.output_jsonl)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_jsonl, 'w') as out:
        for score in scores:
            out.write(json.dumps({'score': score}) + '\n')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_jsonl')
    argp.add_argument('output_jsonl')
    args = argp.parse_args()
    main(args)