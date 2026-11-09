import pickle
import os
import pandas as pd
import pysvs as ps
from multiprocessing import Process
from collections import defaultdict
# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/kabir/similarity-search/dataset/'
pickle_path_1='/data/kabir/similarity-search/models/results_pickles_v7/'
pickle_path_2='/data/kabir/similarity-search/models/results_pickles_v8/'

v8_files=os.listdir(pickle_path_2)
v7_files=os.listdir(pickle_path_1)

def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)


def getStoredData(fileName):
    pickle_path=None
    if fileName in v8_files:
        pickle_path=pickle_path_2
    elif fileName in v7_files:
        pickle_path=pickle_path_1
    if not pickle_path is None:
        with open(pickle_path+fileName, 'rb') as f:
            I = pickle.load(f)
            return I
    return None

def read_vecs_v1(filePath):
    return ps.read_vecs(filePath)

def get_data_generic(dataset):
    # print(dataset)
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs')
    return xb,xq,gt

def refined_recall(gt, I_list, k_gt):
    total_hits = 0
    nq = len(gt)
    total_ids=0
    for q in range(nq):
        gt_set = set(gt[q])
        combined = set()
        for I in I_list:
            combined.update(I[q])
        total_ids=total_ids+len(combined)
        hits = len(gt_set & combined)
        total_hits += hits
    return total_hits / (nq * k_gt), total_ids/(nq * k_gt)

def compute_intersection_fill_recall(gt, I_list, k_gt):
    total_hits = 0
    nq = len(gt)
    n_models = len(I_list)

    for q in range(nq):
        gt_set = set(gt[q])
        candidate_lists = [set(I[q]) for I in I_list]

        # Step 1: intersection of all models
        intersection = set.intersection(*candidate_lists)
        selected = set(intersection)

        if len(selected) < k_gt:
            # Step 2: majority voting for remaining candidates
            votes = defaultdict(int)
            best_rank = {}

            for I in I_list:
                for rank, neighbor in enumerate(I[q]):
                    if neighbor in selected:
                        continue  # skip already selected
                    votes[neighbor] += 1
                    if neighbor not in best_rank:
                        best_rank[neighbor] = rank
                    else:
                        best_rank[neighbor] = min(best_rank[neighbor], rank)

            # Sort by votes (desc), then rank (asc)
            remaining_sorted = sorted(
                votes.keys(),
                key=lambda x: (-votes[x], best_rank[x])
            )

            # Fill until we reach k_gt
            for n in remaining_sorted:
                if len(selected) >= k_gt:
                    break
                selected.add(n)

        # Trim in case we overfilled
        selected = list(selected)[:k_gt]

        hits = len(gt_set & set(selected))
        total_hits += hits

    return total_hits / (nq * k_gt)

def compute_combined_recall(gt, I_list, k_gt):
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return recall1, multiplier, recall2


datasets=[
    'audio',
    'millionSong',
    'yahoomusic_20_variants',
    'agnews-mxbai-1024-euclidean_20_variants',
    'arxiv-nomic-768-normalized_20_variants',
    'ccnews-nomic-768-normalized_20_variants',
    'glove',
    'gooaq-distilroberta-768-normalized_20_variants',
    'deep',
    'imageNet',
    'MNIST',
    'notre',
    'nuswide',
    'sift',
    'sun',
    'laion-clip-512-normalized_20_variants',
    'llama-128-ip_20_variants',
    'coco-nomic-768-normalized_20_variants',
    'yandex-200-cosine_20_variants',
    'yi-128-ip_20_variants'
]


def get_dataset_name_from_file(file_name):
    for dataset in datasets:
        if dataset in file_name:
            return dataset
    return None

file_path = "graph_results_1.csv"
df = pd.read_csv(file_path)
model='FLATNAV'
filtered = df[(df['model'] == model)]
file_names_flatnav = filtered["fileName"].tolist()
file_path = "graph_results_2.csv"
df = pd.read_csv(file_path)
file_names_flatnav.extend(filtered["fileName"].tolist())


for flatnav_result in file_names_flatnav:
    dataset = get_dataset_name_from_file(flatnav_result)
    if not dataset is None:
        flatnav_result=getStoredData(flatnav_result)
        qc=200
        if dataset=='sift':
            qc=10000
        xb,xq,gt=get_data_generic(dataset)
        I1=getStoredData(flatnav_result)
        construction_time_1=I1['construction_time']
        recall1r=I1['recallx'+str(2)]
        recall1=I1['recall']
        search_time1=I1['search_time']
        search_time1r=I1['search_timex2']
        
