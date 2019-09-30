BERT_BASE_DIR=uncased_L-12_H-768_A-12

MENTIONS=$ZESHEL_DATA/mentions
DOCUMENTS=$ZESHEL_DATA/documents
TFIDF_CANDIDATES=$ZESHEL_DATA/tfidf_candidates
OUTPUT_DIR=/tmp/TFRecords/mentions

train_domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft")
val_domains=("coronation_street" "elder_scrolls" "ice_hockey" "muppets")
test_domains=("forgotten_realms" "lego" "star_trek" "yugioh")

concat() {
  documents=""
  for domain in $@; do documents=${documents:+$documents}$DOCUMENTS/${domain}.json,; done
  documents=${documents::-1}
	echo $documents
}

train_documents="$(concat ${train_domains[@]})"
val_documents="$(concat ${val_domains[@]})"
test_documents="$(concat ${test_domains[@]})"
 
mkdir -p $OUTPUT_DIR/{train,val,test}

for split in train heldout_train_seen heldout_train_unseen; do

python create_training_data.py \
  --documents_file=$train_documents \
  --mentions_file=$MENTIONS/${split}.json \
	--tfidf_candidates_file=$TFIDF_CANDIDATES/${split}.json \
  --output_file=$OUTPUT_DIR/train/${split}.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=256 \
  --is_training=True \
  --random_seed=12345 &
done

split="val"

python create_training_data.py \
  --documents_file=$val_documents \
  --mentions_file=$MENTIONS/${split}.json \
	--tfidf_candidates_file=$TFIDF_CANDIDATES/${split}.json \
  --output_file=$OUTPUT_DIR/val \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=256 \
  --is_training=False \
	--split_by_domain=True \
  --random_seed=12345 &

split="test"

python create_training_data.py \
  --documents_file=$test_documents \
  --mentions_file=$MENTIONS/${split}.json \
 	--tfidf_candidates_file=$TFIDF_CANDIDATES/${split}.json \
  --output_file=$OUTPUT_DIR/test \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=256 \
  --is_training=False \
	--split_by_domain=True \
  --random_seed=12345 &
