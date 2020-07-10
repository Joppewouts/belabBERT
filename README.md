# belabBERT 🤧

**Note** the current release of this model is not fully trained yet, the fully trained version of the model will be released later this month

A new Dutch RoBERTa based language model, pretrained on the Dutch unshuffled OSCAR corpus using the masked language modeling (MLM) objective.
The model is case sensitive and includes punctuation. The huggingface🤗  [transformer](https://github.com/huggingface/transformers) library was used for the pretraining process

## Model description

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='jwouts/belabBERT_115k', tokenizer='jwouts/belabBERT_115k')
>>> unmasker("Hoi ik ben een <mask> model.")

[{'sequence': '<s>Hoi ik ben een dames model.</s>',
  'score': 0.05529128015041351,
  'token': 3079,
  'token_str': 'Ġdames'},
 {'sequence': '<s>Hoi ik ben een kleding model.</s>',
  'score': 0.042242035269737244,
  'token': 3333,
  'token_str': 'Ġkleding'},
 {'sequence': '<s>Hoi ik ben een mode model.</s>',
  'score': 0.04132745787501335,
  'token': 6541,
  'token_str': 'Ġmode'},
 {'sequence': '<s>Hoi ik ben een horloge model.</s>',
  'score': 0.029257522895932198,
  'token': 7196,
  'token_str': 'Ġhorloge'},
 {'sequence': '<s>Hoi ik ben een sportief model.</s>',
  'score': 0.028365155681967735,
  'token': 15357,
  'token_str': 'Ġsportief'}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('jwouts/belabBERT_115k')
model = RobertaModel.from_pretrained('jwouts/belabBERT_115k')
text = "Vervang deze tekst."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import RobertaTokenizer, TFRobertaModel
tokenizer = RobertaTokenizer.from_pretrained('jwouts/belabBERT_115k')
model = TFRobertaModel.from_pretrained('jwouts/belabBERT_115k')
text = "Vervang deze tekst."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Release Notes
- Publication of repo: 24 / 06 / 2020
- Publication of model at 150M batches: 10 / 07 / 2020  
- Publication of fully trained model: TBD

## Training data
belabBERT was pretrained on the Dutch version of the **unshuffled** [OSCAR](https://oscar-corpus.com/) corpus, the current state-of-the-art Dutch BERT model [RobBERT](https://github.com/iPieter/RobBERT) was trained on the **shuffled** version of this corpus.
After deduplication the size of this corpus was 32GB

## Training procedure

### Preprocessing

The texts are tokenized using a byte version of Byte-Pair Encoding (BPE) and a vocabulary size of 50.000. The inputs of
the model take pieces of 512 contiguous token that may span over documents. The tokenizer was trained on Dutch texts, The beginning of a new document is marked
with `<s>` and the end of one by `</s>`

The details of the masking procedure for each sentence are the following:
- 20% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

Contrary to BERT, the masking is done dynamically during pretraining (e.g., it changes at each epoch and is not fixed).

### Pretraining

The model was trained on 4 Titan RTX GPUs for 115K steps with a batch size of 1.3K and a sequence length of 512. The
optimizer used is Adam with a learning rate of 5e-5, ![image](https://render.githubusercontent.com/render/math?math=%5Cbeta_%7B1%7D%20%3D%200.9), ![image](https://render.githubusercontent.com/render/math?math=%5Cbeta_%7B2%7D%20%3D%200.98) and
![image](https://render.githubusercontent.com/render/math?math=%5Cepsilon%20%3D%201e%5E%7B-6%7D), a weight decay of 0.01, learning rate warmup for 20000 steps and linear decay of the learning
rate after.

## Evaluation results

Due to credit limitations on the HPC I was not able to finetune belabBERT on the common evaluation tasks.

However, belabBERT is likely to outperform the current state-of-the-art RobBERT since belabBERT uses a Dutch tokenizer where RobBERT is trained with an English tokenizer.
On top of that, RobBERT is trained on a shuffled corpus (at line level) while belabBERT is trained on the unshuffled version of the same corpus, this makes belabBERT more capable to deal with long sequences of text.


## Acknowledgements

This work was carried out on the Dutch national e-infrastructure with the support of [SURF Cooperative](http://surfsara.nl/).

Thanks to the builders of the [OSCAR](https://oscar-corpus.com/) corpus for giving me permission to use the unshuffled Dutch version

A major shout out to the brilliant [@elslooo](https://github.com/elslooo) for the name of this model 🤗

Thanks to the [model card](https://github.com/huggingface/transformers/blob/master/model_cards/roberta-base-README.md) of RoBERTa for the README format/text.
