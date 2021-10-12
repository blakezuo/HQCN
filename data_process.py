import json
import os
import copy
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict

dataset = "aol"
# dataset = "tiangong"

def get_format_train_data():
	filename = os.path.join(dataset, "session_train.txt")
	write_file = os.path.join(dataset, "train.json")
	rf = open(filename)
	wf = open(write_file, 'w')
	line = rf.readline()
	while line:
		sample = json.loads(line.strip())
		sid = str(sample['session_id'])
		history = list()
		for idx, query in enumerate(sample['query']):
			newsample = dict()
			query_text = query['text']
			pos, neg = False, False
			pos_title = ""
			obj = list()
			for idx_doc, click in enumerate(query['clicks']):
				if click['label']:
					if pos:
						continue
					pos_title = click['title']
					newsample['label'] = 1
					pos = True
				else:
					if neg:
						continue
					if pos and click['title'] == pos_title:
						continue
					newsample['label'] = 0
					neg = True
				newsample['query_id'] = sid + '_' + str(idx) + '_' + str(idx_doc)
				newhistory = copy.deepcopy(history)
				newhistory.append({'text':query_text, 'click':click['title']})
				newsample['session'] = newhistory
				obj.append(json.dumps(newsample, ensure_ascii=False))
				if pos and neg:
					break
			if pos and neg and len(obj) == 2:
				for o in obj:
					wf.write(o + '\n')
				history.append({'text':query_text, 'click':pos_title})
		line = rf.readline()

def get_format_test_data():
	filename = os.path.join(dataset, "session_test.txt")
	write_file = os.path.join(dataset, "test.json")
	rf = open(filename)
	wf = open(write_file, 'w')
	line = rf.readline()
	while line:
		sample = json.loads(line.strip())
		sid = str(sample['session_id'])
		history = list()
		for idx, query in enumerate(sample['query']):
			newsample = dict()
			query_text = query['text']
			pos_title = ""
			assert len(query['clicks']) == 50
			max_label = 0
			for idx_doc, click in enumerate(query['clicks']):
				if 'label' in click:
					label = int(click['label'])
				else:
					label = 0
				newhistory = copy.deepcopy(history)
				newsample['query_id'] = sid + '_' + str(idx) + '_' + str(idx_doc)
				newhistory.append({'text':query_text, 'click':click['title'], 'sltb':click['sltb']})
				newsample['session'] = newhistory
				newsample['label'] = label
				jsonObj = json.dumps(newsample, ensure_ascii=False)
				wf.write(jsonObj + '\n')
				if label > max_label:
					pos_title = click['title']
					max_label = label
			history.append({'text':query_text, 'click':pos_title})
		line = rf.readline()

def get_format_test_data_tiangong():
	filename = os.path.join(dataset, "session_test.txt")
	write_file = os.path.join(dataset, "test.json")
	rf = open(filename)
	wf = open(write_file, 'w')
	line = rf.readline()
	while line:
		sample = json.loads(line.strip())
		sid = str(sample['session_id'])
		history = list()
		for idx, query in enumerate(sample['query']):
			newsample = dict()
			query_text = query['text']
			pos_title = ""
			assert len(query['clicks']) == 10
			max_label = 0
			for idx_doc, click in enumerate(query['clicks']):
				if 'label' in click:
					label = int(click['label'])
				else:
					label = 0
				newhistory = copy.deepcopy(history)
				newsample['query_id'] = sid + '_' + str(idx) + '_' + str(idx_doc)
				newhistory.append({'text':query_text, 'click':click['title']})
				newsample['session'] = newhistory
				newsample['label'] = label
				if idx == len(sample['query']) - 1:
					jsonObj = json.dumps(newsample, ensure_ascii=False)
					wf.write(jsonObj + '\n')
				if label > max_label:
					pos_title = click['title']
					max_label = label
			history.append({'text':query_text, 'click':pos_title})
		line = rf.readline()


reform_type = ["generalize", "exploration", "exploiation", "new task"]
def get_reformulation_type():

	def get_type(prev, nexts):
		prev_tokens = prev.strip().split()
		nexts_tokens = nexts.strip().split()
		prevCounter, nextsCounter = Counter(), Counter()
		for token in prev_tokens:
			if token not in stops:
				prevCounter[token] += 1
		for token in nexts_tokens:
			if token not in stops:
				nextsCounter[token] += 1
		common = prevCounter & nextsCounter
		retained = common
		removed = prevCounter - common
		added = nextsCounter - common
		if retained:
			if removed and added:
				return 1
			elif removed:
				return 0
			else:
				return 2
		else:
			return 3

	train_data = "./aol/train.json"
	stopwords = "./stopwords.txt"
	output_file = "./aol/train_reform.json"
	rf = open(stopwords)
	wf = open(output_file,'w')
	stops = dict()
	for line in rf.readlines():
		stops[line.strip()] = True

	rf = open(train_data)
	lines = rf.readlines()
	statistic = [0,0,0,0]
	for line in tqdm(lines):
		sample = json.loads(line.strip())
		prev_query = ""
		for pair in sample['session']:
			query = pair['text']
			type_id = get_type(prev_query, query)
			pair['reform_type'] = reform_type[type_id]
			statistic[type_id] += 1
			prev_query = query
		jsonObj = json.dumps(sample, ensure_ascii=False)
		wf.write(jsonObj + '\n')
	print(statistic)

def get_pair_data():
	rf = open("./aol/train_reform.json")
	wf = open("./aol/train_pair.json", 'w')
	train_set = defaultdict(list)
	for line in tqdm(rf.readlines()):
		sample = json.loads(line.strip())
		query_id = ''.join(sample['query_id'].split('_')[:-1])
		train_set[query_id].append(sample)
	for qid in train_set:
		newsample = dict()
		samplelist = train_set[qid]
		pos_list = [v for v in samplelist if v['label']]
		neg_list = [v for v in samplelist if not v['label']]
		if not neg_list or not pos_list:
			continue
		pl, nl = len(pos_list), len(neg_list)
		newsample['query_id'] = qid
		newsample['history'] = pos_list[0]['session'][:-1]
		newsample['current_query'] = pos_list[0]['session'][-1]['text']
		newsample['reform_type'] = pos_list[0]['session'][-1]['reform_type']
		for i in range(max(pl, nl)):
			newsample['pos_candidate'] = pos_list[i%pl]['session'][-1]['click']
			newsample['pos_features'] = pos_list[i%pl]['session'][-1]['aol']
			newsample['neg_candidate'] = neg_list[i%nl]['session'][-1]['click']
			newsample['neg_features'] = neg_list[i%nl]['session'][-1]['aol']
			jsonObj = json.dumps(newsample, ensure_ascii=False)
			wf.write(jsonObj + '\n')

def get_long_data():
	rf = open("./aol/train_pair.json")
	wf = open("./aol/train_long.json", 'w')
	for line in rf.readlines():
		sample = json.loads(line.strip())
		if len(sample['history']) < 1:
			continue
		wf.write(line)

def get_long_data_test():
	rf = open("./aol/test_reform.json")
	wf = open("./aol/test_long.json", 'w')
	for line in rf.readlines():
		sample = json.loads(line.strip())
		if len(sample['session']) < 2:
			continue
		wf.write(line)

# get_training data
get_format_train_data()
get_reformulation_type()
get_pair_data()
get_long_data()
