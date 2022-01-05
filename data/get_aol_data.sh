# echo "get_format_train_data..."
# python data_process.py aol train get_format_train_data
# echo "get_reformulation_type..."
# python data_process.py aol train get_reformulation_type
# echo "get_pair_data..."
# python data_process.py aol train get_pair_data
# rm ./aol/train_format.json
# rm ./aol/train.json

# echo "get_format_test_data..."
# python data_process.py aol test get_format_test_data
# echo "get_reformulation_type..."
# python data_process.py aol test get_reformulation_type
# rm ./aol/test_format.json

echo "get_features..."
python get_features.py aol