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

## Code: Coming soon..

## Reference
If you use our dataset in your work, please cite us.

*Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin and Honglak Lee. 2019. [Zero-Shot Entity Linking by Reading Entity Descriptions](https://www.aclweb.org/anthology/P19-1335). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.*
```
@inproceedings{logeswaran-etal-2019-zero,
    title = "Zero-Shot Entity Linking by Reading Entity Descriptions",
    author = "Logeswaran, Lajanugen and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina and Devlin, Jacob and Lee, Honglak",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019"
}
```
