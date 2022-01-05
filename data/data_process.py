import json
import os
import copy
import sys
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict

def get_format_train_data():
	filename = os.path.join(dataset, "session_train.txt")
	write_file = os.path.join(dataset, "train_format.json")
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
			pos_title = []
			obj = list()
			for idx_doc, click in enumerate(query['clicks']):
				if 'label' in click and click['label']:
					pos_title.append(click['title'])
					newsample['label'] = 1
				else:
					if click['title'] in pos_title:
						continue
					newsample['label'] = 0
				newsample['query_id'] = sid + '_' + str(idx) + '_' + str(idx_doc)
				newhistory = copy.deepcopy(history)
				newhistory.append({'text':query_text, 'click':click['title']})
				newsample['session'] = newhistory
				jsonObj = json.dumps(newsample, ensure_ascii=False)
				wf.write(jsonObj + '\n')
			if not pos_title:
				history.append({'text':query_text, 'click':""})
			else:
				history.append({'text':query_text, 'click':pos_title[0]})
		line = rf.readline()

def get_format_test_data():
	filename = os.path.join(dataset, "session_test.txt")
	write_file = os.path.join(dataset, "test_format.json")
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
				newhistory.append({'text':query_text, 'click':click['title']})
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
	write_file = os.path.join(dataset, "test_format.json")
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

	train_data = os.path.join(dataset, f"{mode}_format.json")
	stopwords = "./stopwords.txt"
	output_file = os.path.join(dataset, f"{mode}.json")
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
		if len(sample['session']) < 2:
			continue
		prev_query = ""
		for pair in sample['session']:
			query = pair['text']
			type_id = get_type(prev_query, query)
			pair['reform_type'] = reform_type[type_id]
			statistic[type_id] += 1
			prev_query = query
		jsonObj = json.dumps(sample, ensure_ascii=False)
		wf.write(jsonObj + '\n')

def get_pair_data():
	rf = open(os.path.join(dataset, f"{mode}.json"))
	wf = open(os.path.join(dataset, f"{mode}_pair.json"), 'w')
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
			newsample['neg_candidate'] = neg_list[i%nl]['session'][-1]['click']
			jsonObj = json.dumps(newsample, ensure_ascii=False)
			wf.write(jsonObj + '\n')

if __name__ == "__main__":
	dataset = sys.argv[1]
	mode = sys.argv[2]
	globals()[sys.argv[3]]()

# get_format_train_data()
# get_reformulation_type()
# get_pair_data()
