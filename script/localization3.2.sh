date "+%Y-%m-%d %H:%M:%S"
python agentless/fl/localize.py \
    --merge \
    --output_folder results/$FOLDER_NAME/edit_location_individual \
    --top_n 7 \
    --num_samples $NUM_SETS \
    --start_file results/$FOLDER_NAME/edit_location_samples/loc_outputs.jsonl \
    ${TARGET_ID:+--target_id $TARGET_ID}

