# Zero-shot Entity Linking

The data is available [here](https://drive.google.com/file/d/1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO/view?usp=sharing). 

## License
This dataset was constructed using [Wikias](https://community.fandom.com/wiki/Hub:Big_wikis) from [FANDOM](https://www.fandom.com) and is licensed under the Creative Commons Attribution-Share Alike License (CC-BY-SA).

## Data 

#### Entity Dictionary
Documents/pages from the wikias serve as both context for mentions and entity descriptions. Each document corresponds to an entity and the collection of documents comprise the entity dictionary. The documents are organized as `documents/<wikia>.json`, each line of which represents a document/entity from the corresponding wikia, and has the following format.

```
{"document_id": "000523A4D586C293", "title": "Warner", "text": "Warner Warner was a communications technician aboard Nerva Beacon ..."}
{"document_id": "0009247003C7CB16", "title": "Winnie Tyler", "text": "Winnie Tyler Winnie Tyler was Jacob Tyler ' s wife ..."}
```
Description of fields:
* `document_id`: Unique identifier of document/entity
* `title`: Document title
* `text`: Document content/Text description of entity

#### Mentions
The mentions are organized into the following files: `{train,heldout_train_seen,heldout_train_unseen,val,test}.json`. Each file has the following format. 
```
{"mention_id": "A67C0D65787A42DB", "context_document_id": "88798F9F1493965A", "corpus": "starwars", "start_index": 4085, "end_index": 4085, "text": "prevented", "label_document_id": "CD37AAA25C04F62D", "category": "LOW_OVERLAP"}
{"mention_id": "B7C96945B4EB593F", "context_document_id": "F2845988E1AC5BC0", "corpus": "military", "start_index": 178, "end_index": 182, "text": "C . W . Jessen", "label_document_id": "33B704F9C31B1D57", "category": "LOW_OVERLAP"}
```

Description of fields:
* `mention_id`: Unique mention identifier
* `context_document_id`: Identifier of mention's source document
* `start_index, end_index`: Location of mention text in the source document, assuming white-space tokenization (0-based indexing, start and end positions inclusive)
* `text`: Mention phrase
* `label_document_id`: Document id of entity represented by the mention
* `category`: Type of mention 
* `corpus`: Source world of the mention

## Code

This code is largely based on the [BERT codebase](https://github.com/google-research/bert). The BERT codebase is also a dependency, so please clone the BERT repository to use this code.

### Entity Linking

#### Data Preparation

Data needs to be converted to TFRecords for training and evaluation. The script `scripts/create_training_data.sh` is provided for this purpose. Set the `ZESHEL_DATA` variable to the path of the downloaded data.

In addition to the mentions provided in the data, the output of the candidate generation phase will also be necessary to run this script. We provide [top-64 candidates](https://drive.google.com/file/d/1wGppj3ivE7jBaDzDlovWAvaBzLhPjR8B/view?usp=sharing)  generated using TFIDF as described in the paper. Move the `tfidf_candidates` folder to the data directory before running the script.

Once the TFRecords have been generated, we can now train and evaluate models.

#### Training

Use `scripts/train_classifier.sh` to train an Entity Linking model using the training data. The provided script finetunes a BERT-base model for Entity Linking. Set `BERT_BASE_DIR` to the path of a downloaded pre-trained BERT-base model.

Other training settings in the paper can be reproduced by initializing the model with appropriate pre-trained models using the above script in an analogous manner.

#### Evaluation

Use `scripts/eval_classifier.sh` to evaluate the normalized Entity Linking performance of a trained Entity Linking model. Set the `domain` variable to an appropriate split (Eg: `train/heldout_train_unseen`, `val/coronation_street`, `test/forgotten_realms`, ...).

### Pre-training

The script `scripts/create_pretraining_data.sh` prepares TFRecords for Language Model pre-training.

The paper explores many pre-training recipes. We provide one particular pre-training recipe in `scripts/run_pretraining.sh` (Open-corpus pre-training -> Task-adaptive pre-training -> Domain-adaptive pre-training).

Note: Models is the paper were trained using TPUs. Some hyperparameters (Eg: Batch size, context size, number of entity candidates, etc.) may have to be changed for models to fit in GPUs.

## Reference
If you use our dataset/code in your work, please cite us.

*Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin and Honglak Lee. 2019. [Zero-Shot Entity Linking by Reading Entity Descriptions](https://www.aclweb.org/anthology/P19-1335). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.*
```
@inproceedings{logeswaran-etal-2019-zero,
    title = "Zero-Shot Entity Linking by Reading Entity Descriptions",
    author = "Logeswaran, Lajanugen and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina and Devlin, Jacob and Lee, Honglak",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019"
}
```
