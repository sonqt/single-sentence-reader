# Single-Sentence Reader: A Novel Approach for Addressing Answer Position Bias
![An Illustration of the Inner Workings of Single-Sentence Reader](images/pipeline.png)
## Introduction
This repository contains the source code for the architectures described in the following paper:
>**Single-Sentence Reader: A Novel Approach for Addressing Answer Position Bias**

>Son Quoc Tran, Matt Kretchmar

>Computer Science Department, Denison University, Granville, Ohio

Machine Reading Comprehension (MRC) models tend to take advantage of spurious correlations (also known as dataset bias or annotation artifacts in the research community). Consequently, these models may perform the MRC task without fully comprehending the given context and question, which is undesirable since it may result in low robustness against distribution shift. This paper delves into the concept of answer-position bias, where a significant percentage of training questions have answers located solely in the first sentence of the context. We propose a Single-Sentence Reader as a new approach for addressing answer position bias in MRC. We implement this approach using six different models and thoroughly analyze their performance. Remarkably, our proposed Single-Sentence Readers achieve results that nearly match those of models trained on conventional training sets, proving their effectiveness. Our study also discusses several challenges our Readers encounter and proposes a potential solution. 

---
## Getting Started
This codecase uses Python 3.9.12
Dependencies:
```
conda create -n ssent -y python=3.9.12
conda activate ssent
pip install -r requirements.txt
```
## Preparing Data
### SQuAD
Data was originates from SQuAD version 1.1. For further details, please refer to the [SQuAD explorer](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset). 
Save the downloaded train and dev sets in `/Data`
### Get Biased Data
This part is partially adopted from [`KazutoshiShinoda/ShortcutLearnability`](https://github.com/kazutoshishinoda/shortcutlearnability).
```
python src/position_bias.py <path_to_dataset> <analysis_save_path>
```

### Get Single-Sentence Training Data

### Decontextualize SQuAD
Use notebook `src/decontextualize_SQuAD.ipynb`
## Training and Predicting
This section is based on the code examples from [huggingface/transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

Train
```
TRAIN_PATH=""
SAVE_PATH=""
model="bert-base-cased"
python code/run_qa.py \
    --model_name_or_path ${model} \
    --train_file "${TRAIN_PATH}" \
    --do_train \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --max_answer_length 32 \
    --doc_stride 128 \
    --save_steps 999999 \
    --overwrite_output_dir \
    --version_2_with_negative \     # Delete this if no unanswerable
    --output_dir "${SAVE_PATH}/${model}"
```

Predict
```
MODEL_PATH=""
EVAL_PATH=""
SAVE_PATH=""

model="bert-base-cased"
python code/run_qa.py \
    --model_name_or_path "${MODEL_PATH}/${model}" \
    --validation_file "${EVAL_PATH}" \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --max_seq_length 128 \
    --max_answer_length 32 \
    --doc_stride 32 \
    --n_best_size 5 \
    --overwrite_output_dir \
    --version_2_with_negative \         # Delete this if no unanswerable
    --output_dir "${SAVE_PATH}/${model}"
```
### Evaluate Single-Sentence Reader
```
code/evaluate_single-sentence.py <dataset_file> <prediction_file>
```
***Note:*** `dataset_file` mentioned here is a JSON file formatted for use with `code/run_qa.py`. It is the same file used for prediction in the previous part.

---

### Acknowledgement
This codebase incorporates numerous public materials contributed by other researchers (refer to each section for detailed information). We wholeheartedly express our gratitude to these researchers for their invaluable contributions to the advancement of NLP.

We also want to thank Krystal Ly for contributing to this project in its initial phase, and Antony Silveira for his support with technical issues.
## Citation and Contact
```
Arxiv Bibtex Here
```
Please contact Son Quoc Tran at `tran_s2[at]denison.edu` and Matt Kretchmar at `kretchmar@denison.edu` if you have any questions.
