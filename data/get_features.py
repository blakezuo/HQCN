import json
import numpy as np
from collections import defaultdict
from collections import Counter
from numpy import linalg as la
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import math
import time
import os
import sys
from tqdm import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity

def get_idf_file(file_list, outfile):
	word_dict = defaultdict(float)
	doc_num = 0
	for filename in file_list:
		rf = open(filename, 'r', encoding='utf-8')
		line = rf.readline()
		while line:
			sample = json.loads(line.strip())
			for query in sample['query']:
				for click in query['clicks']:
					d_terms = click['title'].split()
					for term in set(d_terms):
						word_dict[term] += 1
					doc_num += 1
			line = rf.readline()
	for word in word_dict:
		word_dict[word] = 1.0 * doc_num / (word_dict[word] + 1)
	wf = open(outfile, 'w', encoding='utf-8')
	json.dump(word_dict, wf)

def read_embedding(file):
	embedding = defaultdict(list)
	embedding['[pad]'] = np.random.rand(d_word)
	embedding['[unk]'] = np.zeros(d_word)
	rf = open(file, 'r', encoding='utf-8')
	line = rf.readline()
	while line:
		cut = line.strip().split()
		line = rf.readline()
		if len(cut) != d_word + 1:
			continue
		try:
			embedding[cut[0]] = np.array([float(x) for x in cut[1:]])
		except:
			continue
	return embedding

def get_tfidf(word_list):
	tfidf = []
	for word in word_list:
		tf = 1.0 * word_list.count(word) / len(word_list)
		if word in term_dict.keys():
			idf = max(0,math.log(term_dict[word]))
		else:
			idf = 0.0
		tfidf.append(tf * idf)
	return tfidf

def euclidean_distance(vector1, vector2): #欧式距离
	return pdist(np.vstack((vector1, vector2)), 'euclidean')[0]

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def add_feature_train(infile, outfile):

	def add_self_feats(terms):
		if len(terms) == 0:
			return [0.0]
		tfidfs = get_tfidf(terms)
		ave = 1.0 * sum(tfidfs) / len(tfidfs)
		return [ave]

	def add_pair_feats(q_terms, d_terms):
		com = 0.0
		for term in d_terms:
			if term in q_terms:
				com += 1
		if len(d_terms) == 0:
			com = 0.0
		else:
			com = com / len(d_terms)

		pad = [0.0 for _ in range(d_word)]
		q_rep = np.zeros(d_word, np.float)
		for t in q_terms:
			q_rep = q_rep + np.asarray(emb_dict.get(t, pad))
		d_rep = np.zeros(d_word, np.float)
		for t in d_terms:
			d_rep = d_rep + np.asarray(emb_dict.get(t, pad))
		sum_sim = cos_sim(q_rep, d_rep)

		try:
			q_tfidf = get_tfidf(q_terms)
			max_term = q_terms[q_tfidf.index(max(q_tfidf))]
			q_rep_max = emb_dict.get(max_term, pad)
			d_tfidf = get_tfidf(d_terms)
			max_term = d_terms[d_tfidf.index(max(d_tfidf))]
			d_rep_max = emb_dict.get(max_term, pad)
			max_sim = cos_sim(q_rep_max, d_rep_max)
		except:
			max_sim = 0.0
		return [com, sum_sim, max_sim]


	def add_his_feats(hq_terms, hc_terms, d_terms):
		com, sum_sim, max_sim = 0.0, 0.0, 0.0
		for q_terms in hq_terms:
			feats = add_pair_feats(q_terms, d_terms)
			com += feats[0]
			sum_sim += feats[1]
			max_sim += feats[2]
		com = com / len(hq_terms)
		sum_sim = sum_sim / len(hq_terms)
		max_sim = max_sim / len(hq_terms)

		hc, hs, hm, appear = 0.0, 0.0, 0.0, 0
		for c_terms in hc_terms:
			feats = add_pair_feats(c_terms, d_terms)
			hc += feats[0]
			hs += feats[1]
			hm += feats[2]
			if " ".join(d_terms) == " ".join(c_terms):
				appear = 1

		hc = hc / len(hc_terms)
		hs = hs / len(hc_terms)
		hm = hm / len(hc_terms)
		return [com, sum_sim, max_sim, hc, hs, hm, appear]
	
	def add_click_repeat_feats(d_terms, hc_terms):
		def get_sentence_rep(terms):
			pad = [0.0 for _ in range(d_word)]
			rep = np.zeros(d_word, np.float)
			for term in terms:
				rep += np.asarray(emb_dict.get(term, pad))
			return rep
		
		d_rep = get_sentence_rep(d_terms)
		max_sim = 0.0
		for terms in hc_terms:
			h_rep = get_sentence_rep(terms)
			max_sim = max(max_sim, cos_sim(d_rep, h_rep))
		return [max_sim]

	def get_feats(q_terms, d_terms, hq_terms, hc_terms):
		features = list()
		features.extend(add_self_feats(d_terms))
		features.extend(add_pair_feats(q_terms, d_terms))
		features.extend(add_his_feats(hq_terms, hc_terms, d_terms))
		features.extend(add_click_repeat_feats(d_terms, hc_terms))
		return features


	rf = open(infile, 'r', encoding = 'utf-8')
	wf = open(outfile, 'w', encoding='utf-8')
	for line in tqdm(rf.readlines()):
		sample = json.loads(line.strip())
		hq_terms, hc_terms = list(), list()
		for his in sample['history']:
			hq_terms.append(his['text'].split())
			hc_terms.append(his['click'].split())
		q_terms = sample['current_query'].split()
		pd_terms = sample['pos_candidate'].split()
		nd_terms = sample['neg_candidate'].split()

		sample['pos_features'] = get_feats(q_terms, pd_terms, hq_terms, hc_terms)
		sample['neg_features'] = get_feats(q_terms, nd_terms, hq_terms, hc_terms)
		assert len(sample['pos_features']) == 12
		assert len(sample['neg_features']) == 12
		jsonObj = json.dumps(sample, ensure_ascii=False)
		wf.write(jsonObj)
		wf.write('\n')


def add_feature_test(infile, outfile):

	def add_self_feats(terms):
		if len(terms) == 0:
			return [0.0]
		tfidfs = get_tfidf(terms)
		ave = 1.0 * sum(tfidfs) / len(tfidfs)
		return [ave]

	def add_pair_feats(q_terms, d_terms):
		com = 0.0
		for term in d_terms:
			if term in q_terms:
				com += 1
		if len(d_terms) == 0:
			com = 0.0
		else:
			com = com / len(d_terms)

		pad = [0.0 for _ in range(d_word)]
		q_rep = np.zeros(d_word, np.float)
		for t in q_terms:
			q_rep = q_rep + np.asarray(emb_dict.get(t, pad))
		d_rep = np.zeros(d_word, np.float)
		for t in d_terms:
			d_rep = d_rep + np.asarray(emb_dict.get(t, pad))
		sum_sim = cos_sim(q_rep, d_rep)

		try:
			q_tfidf = get_tfidf(q_terms)
			max_term = q_terms[q_tfidf.index(max(q_tfidf))]
			q_rep_max = emb_dict.get(max_term, pad)
			d_tfidf = get_tfidf(d_terms)
			max_term = d_terms[d_tfidf.index(max(d_tfidf))]
			d_rep_max = emb_dict.get(max_term, pad)
			max_sim = cos_sim(q_rep_max, d_rep_max)
		except:
			max_sim = 0.0
		return [com, sum_sim, max_sim]


	def add_his_feats(hq_terms, hc_terms, d_terms):
		com, sum_sim, max_sim = 0.0, 0.0, 0.0
		for q_terms in hq_terms:
			feats = add_pair_feats(q_terms, d_terms)
			com += feats[0]
			sum_sim += feats[1]
			max_sim += feats[2]
		com = com / len(hq_terms)
		sum_sim = sum_sim / len(hq_terms)
		max_sim = max_sim / len(hq_terms)

		hc, hs, hm, appear = 0.0, 0.0, 0.0, 0
		for c_terms in hc_terms:
			feats = add_pair_feats(c_terms, d_terms)
			hc += feats[0]
			hs += feats[1]
			hm += feats[2]
			if " ".join(d_terms) == " ".join(c_terms):
				appear = 1

		hc = hc / len(hc_terms)
		hs = hs / len(hc_terms)
		hm = hm / len(hc_terms)
		return [com, sum_sim, max_sim, hc, hs, hm, appear]
	
	def add_click_repeat_feats(d_terms, hc_terms):
		def get_sentence_rep(terms):
			pad = [0.0 for _ in range(d_word)]
			rep = np.zeros(d_word, np.float)
			for term in terms:
				rep += np.asarray(emb_dict.get(term, pad))
			return rep
		
		d_rep = get_sentence_rep(d_terms)
		max_sim = 0.0
		for terms in hc_terms:
			h_rep = get_sentence_rep(d_terms)
			max_sim = max(max_sim, cos_sim(d_rep, h_rep))
		return [max_sim]

	def get_feats(q_terms, d_terms, hq_terms, hc_terms):
		features = list()
		features.extend(add_self_feats(d_terms))
		features.extend(add_pair_feats(q_terms, d_terms))
		features.extend(add_his_feats(hq_terms, hc_terms, d_terms))
		features.extend(add_click_repeat_feats(d_terms, hc_terms))
		return features


	rf = open(infile, 'r', encoding = 'utf-8')
	wf = open(outfile, 'w', encoding='utf-8')
	for line in tqdm(rf.readlines()):
		sample = json.loads(line.strip())
		hq_terms, hc_terms = list(), list()
		for his in sample['session'][:-1]:
			hq_terms.append(his['text'].split())
			hc_terms.append(his['click'].split())
		q_terms = sample['session'][-1]['text'].split()
		d_terms = sample['session'][-1]['click'].split()

		sample['features'] = get_feats(q_terms, d_terms, hq_terms, hc_terms)
		jsonObj = json.dumps(sample, ensure_ascii=False)
		wf.write(jsonObj)
		wf.write('\n')


path = sys.argv[1]
d_word = 256
train_file = os.path.join(path, 'session_train.txt')
test_file = os.path.join(path, 'session_test.txt')
get_idf_file([train_file, test_file], os.path.join(path, "idf.json"))

term_dict = json.load(open(os.path.join(path, "idf.json"),'r',encoding='utf-8'), strict=False)
print("loading word_embedding...", time.asctime(time.localtime(time.time())))
emb_dict = read_embedding(os.path.join(path, 'fasttext.model'))
print(len(emb_dict))
print("get features...", time.asctime(time.localtime(time.time())))

add_feature_train(os.path.join(path, "train_pair.json"), os.path.join(path, "train_feature.json"))
add_feature_test(os.path.join(path, "test.json"), os.path.join(path, "test_feature.json"))