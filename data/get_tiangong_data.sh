echo "get_format_train_data..."
python data_process.py tiangong train get_format_train_data
echo "get_reformulation_type..."
python data_process.py tiangong train get_reformulation_type
echo "get_pair_data..."
python data_process.py tiangong train get_pair_data
rm ./tiangong/train_format.json
rm ./tiangong/train.json

echo "get_format_test_data..."
python data_process.py tiangong test get_format_test_data_tiangong
echo "get_reformulation_type..."
python data_process.py tiangong test get_reformulation_type
rm ./tiangong/test_format.json

echo "get_features..."
python get_features.py tiangong