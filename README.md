# Improving Session Search by Modeling Multi-Granularity Historical Query Change
Xiaochen Zuo, Zhicheng Dou, Ji-Rong Wen

This repo provides code, trained models for our WSDM Full paper [Improving Session Search by Modeling Multi-Granularity Historical Query Change].

## Requirements
pytorch-transformers==1.2.0
pytrec-eval==0.5
torch==1.6.0

## Data Download
To download all the needed data, run:
```
bash download_data.sh (will be added in the future)
```

## Data Preprocess
To get preprocessed AOL data, run:
```
cd data
bash get_aol_data.sh
```
To get preprocessed Tiangong-ST data, run:
```
bash get_tiangong_data.sh
```