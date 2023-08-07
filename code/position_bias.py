import json
import sys
import os
import spacy

def load_json(file):
    with open(file, 'r') as f:
        d = json.load(f)
    return d

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
def get_offset_id2w(doc):
    # from char index to word index
    id2w = {}
    tm1 = 0
    for i, t in enumerate(doc):
        for j in range(tm1, t.idx):
            # t.idx: char-level start position of t (token)
            id2w[j] = i - 1
        tm1 = t.idx
    for j in range(t.idx, t.idx+len(t)+1):
        id2w[j] = i
    return id2w
def process_qa_data(raw_data):
    ds_data = []

    dem = 0
    for a in raw_data['data']:
        for p in a['paragraphs']:
            for qa in p['qas']:
                id = qa['id']
                dem += 1
                answer_start, answer_text = [], []
                for an in qa['answers']:
                    answer_start.append(an['answer_start'])
                    answer_text.append(an['text'])
                ds_data.append({
                    "id": id,
                    "context": p['context'],
                    "question": qa['question'],
                    "answers": {'answer_start': answer_start, 'text': answer_text}})
    return ds_data
def main(path_to_data, save_path):
    raw_data = load_json(path_to_data)
    data = process_qa_data(raw_data)

    results_dict = {'answer-position-sentence': {}}
    nlp = spacy.load("en_core_web_sm")
    total = 0
    biased = 0

    for example in data:
        if total % 100 == 0:
            print(total)
        total += 1
        #Rewrite this to new_format
        _id = example['id']
        c = example['context']
        q = example['question']
        a = example['answers']['text'][0]

        astart = example['answers']['answer_start'][0]
        aend = astart + len(a) - 1
        try:
            doc = nlp(c)
        except:
            print(f"Parsing context error: {c}")
            continue
        id2w = get_offset_id2w(doc)
        aw_start = id2w[astart]
        aw_end = id2w[aend] + 1

        # Find the answer and save to dictionary
        for i, sent in enumerate(doc.sents):
            if sent.start <= aw_start and aw_end <= sent.end:
                results_dict['answer-position-sentence'][_id] = i
                if i == 0:
                    biased += 1
                break
        if not _id in results_dict['answer-position-sentence']:
            results_dict['answer-position-sentence'][_id] = None
    print("{} percent of samples are biased!".format(biased/total*100))
    # Save result here
    save_json(results_dict['answer-position-sentence'], save_path)
    # save_json(results_dict['answer-position-sentence'], save_path)
    # return results_dict
if __name__ == '__main__':
    path_to_data = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    main(path_to_data, save_path)
