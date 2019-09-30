BERT_BASE_DIR=uncased_L-12_H-768_A-12
INPUT_DIR=$ZESHEL_DATA/documents
OUTPUT_DIR=/tmp/TFRecords/documents

mkdir -p $OUTPUT_DIR

for input_file in $INPUT_DIR/*; do
  fname=${input_file##*/}
  domain=${fname%.json}
	echo $domain
  python create_pretraining_data.py \
    --input_file=$input_file \
    --output_file=$OUTPUT_DIR/${domain}.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=256 \
    --random_seed=12345
done

mkdir -p $OUTPUT_DIR/{train,val,test}

train_domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft")
val_domains=("coronation_street" "elder_scrolls" "ice_hockey" "muppets")
test_domains=("forgotten_realms" "lego" "star_trek" "yugioh")

for domain in ${train_domains[@]}; do mv $OUTPUT_DIR/${domain}.tfrecord $OUTPUT_DIR/train/; done
for domain in ${val_domains[@]};   do mv $OUTPUT_DIR/${domain}.tfrecord $OUTPUT_DIR/val/; done
for domain in ${test_domains[@]};  do mv $OUTPUT_DIR/${domain}.tfrecord $OUTPUT_DIR/test/; done
