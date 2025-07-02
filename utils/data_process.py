import math

# from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import torch
import random
# import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
import os
from tqdm import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import openai
from multiprocessing import Process
import scipy.sparse as sp
# from multiprocessing import  pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops, add_self_loops
import copy
import json
from tqdm import tqdm
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN

"""
this function is for getting node sequence around  mode [node_idx], use avoid_idx for link prediction task to filter the other node
"""
def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence

def generate_nc_jsonl(data_name='pubmed', k_hop=2, sample_size=10):
    seed = 435
    np.random.seed(seed)
    random.seed(seed)
    data = torch.load(os.path.join('dataset',data_name,'processed_data.pt'), weights_only=False)
    edge_list = generate_edge_list(data)  # obtain the list of 1-hop sub-graphs
    # TODO: node_idx
    node_set = torch.arange(data.edge_index.max()).numpy()
    if data_name == 'pubmed':
        train_ratio, val_ratio = 0.6, 0.2
        qs = f" Given a node-centered graph:{DEFAULT_GRAPH_TOKEN}, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to?"
    elif data_name == 'cora':
        train_ratio, val_ratio = 0.6, 0.2
        qs = f" Given a node-centered graph:{DEFAULT_GRAPH_TOKEN}, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
    elif data_name == 'products':
        train_ratio, val_ratio = 0.08, 0.02
        qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
    elif data_name == 'arxiv':
        train_ratio, val_ratio = 6/11, 2/11
        qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
    human_conv = {"from": "human", "value": qs}
    gpt_conv = {"from": "gpt", "value": None}

    total_size = node_set.shape[0]
    train_size = round(total_size * train_ratio)
    val_size = round(total_size * val_ratio)
    # test_size = total_size - train_size - val_size

    np.random.shuffle(node_set)
    train_idx = np.sort(node_set[:train_size]).tolist()
    val_idx = np.sort(node_set[train_size: train_size+val_size]).tolist()
    test_idx = np.sort(node_set[train_size+val_size:]).tolist()

    jsonl_train_file = f"sampled_{k_hop}_{sample_size}_train_zx.jsonl"
    jsonl_val_file = f"sampled_{k_hop}_{sample_size}_val_zx.jsonl"
    jsonl_test_file = f"sampled_{k_hop}_{sample_size}_test_zx.jsonl"

    with open(os.path.join("./dataset",data_name, jsonl_train_file), 'w') as f:
        print('Prepare the train jsonl file... \n')
        for node_idx in tqdm(train_idx):
            sample_dic = {}
            node_sequence = get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size)
            sample_dic['id'] = node_idx
            sample_dic['graph'] = node_sequence
            gpt_conv['value'] = data.label_texts[data.y[node_idx]]
            sample_dic['conversations'] = [human_conv, gpt_conv]
            json_line = json.dumps(sample_dic)
            f.write(json_line + '\n')
    
    with open(os.path.join("./dataset",data_name, jsonl_val_file), 'w') as f:
        print('Prepare the val jsonl file... \n')
        for node_idx in tqdm(val_idx):
            sample_dic = {}
            node_sequence = get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size)
            sample_dic['id'] = node_idx
            sample_dic['graph'] = node_sequence
            gpt_conv['value'] = data.label_texts[data.y[node_idx]]
            sample_dic['conversations'] = [human_conv, gpt_conv]
            json_line = json.dumps(sample_dic)
            f.write(json_line + '\n')
    
    with open(os.path.join("./dataset",data_name, jsonl_test_file), 'w') as f:
        print('Prepare the test jsonl file... \n')
        for node_idx in tqdm(test_idx):
            sample_dic = {}
            node_sequence = get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size)
            sample_dic['id'] = node_idx
            sample_dic['graph'] = node_sequence
            gpt_conv['value'] = data.label_texts[data.y[node_idx]]
            sample_dic['conversations'] = [human_conv, gpt_conv]
            # print(sample_dic)
            json_line = json.dumps(sample_dic)
            f.write(json_line + '\n')
            
"""
get edge_list from pyg edge_index\
"""
def generate_edge_list(data):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    row, col = data.edge_index
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    # torch.save(edge_list, os.path.join(data_dir, "edge_list.pt"))
    return edge_list

from torch_geometric.utils import k_hop_subgraph
class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def partition_propagate(self, data_edge_index, x, norm, select_idx=None, chunk_size=800, cuda=False):
        if select_idx is None:
            n = x.shape[0]
            select_idx = torch.arange(n)
        else:
            n = select_idx.shape[0]

        os=[]
        for i in trange(0, n, chunk_size):
            key=select_idx[i:i+chunk_size]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(key, 1, data_edge_index, relabel_nodes=True)
            if cuda:
                o =  self.propagate(edge_index.cuda(), x=x[subset].cuda(), norm=norm[edge_mask].cuda())
            else:
                o = self.propagate(edge_index, x=x[subset], norm=norm[edge_mask])
            os.append(o[mapping])

        return torch.cat(os, dim=0)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def generate_notestlink(dataset):
    data_dir = f"dataset/{dataset}"
    data = torch.load(f"dataset/{dataset}/processed_data.pt")
    print(data)
    useless_keys = ['val_id', 'test_id', 'title', 'abs', 'train_id', 'label_texts', 'raw_texts', 'keywords',
                    'category_names', 'label_names']
    for k in useless_keys:
        if k in data:
            data[k] = None
    useful_keys = ['train_mask', 'x', 'val_mask', 'edge_index', 'test_mask', 'y']
    for k in useful_keys:
        if k in data:
            data[k] = data[k].contiguous()
    link_test_path = os.path.join(data_dir, "edge_sampled_2_10_only_test.jsonl")
    with open(link_test_path, 'r') as f:
        link_test_lines = f.readlines()
        link_test_lines = [json.loads(line) for line in link_test_lines]
        test_links = [tuple(line['id']) for line in link_test_lines if line["conversations"][1]["value"] == "yes"]
    links = set(test_links)
    new_edge_index = []
    old_edge_index = data.edge_index.numpy().tolist()
    remove=1
    for i in trange(len(old_edge_index[0])):
        if (old_edge_index[0][i], old_edge_index[1][i]) in links or (old_edge_index[1][i], old_edge_index[0][i]) in links:
            remove+=1
            continue
        else:
            new_edge_index.append([old_edge_index[0][i], old_edge_index[1][i]])

    new_edge_index = torch.LongTensor(new_edge_index).t()
    data.edge_index = new_edge_index.contiguous()
    torch.save(data,f"dataset/{dataset}/processed_data_link_notest.pt")


def generate_multi_hop_x_arxiv_notestlink(emb="sbert"):
    data = torch.load(f"dataset/ogbn-arxiv/processed_data_link_notest.pt")
    x = torch.load(f"dataset/ogbn-arxiv/{emb}_x.pt")
    edge_index = data.edge_index
    row, col = data.edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    link_test_path = os.path.join(f"dataset/ogbn-arxiv", "edge_sampled_2_10_only_test.jsonl")
    with open(link_test_path, 'r') as f:
        link_test_lines = f.readlines()
    link_test_lines = [json.loads(line) for line in link_test_lines]
    n = data.num_nodes
    mask = torch.full([n], fill_value=False, dtype=torch.bool)
    for link in link_test_lines:
        mask[link['id'][0]] = True
        mask[link['id'][1]] = True
    mp = MP()
    torch.save(mask, f"dataset/ogbn-arxiv/no_test_link_mask.pt")
    for i in range(4):
        x = mp.propagate(edge_index, x=x, norm=norm)
        torch.save(x[mask].cpu(), f"dataset/ogbn-arxiv/{emb}_{i + 1}hop_x_notestlink.pt")



def generate_multi_hop_x_products_notestlink(emb="sbert"):
    print(emb)
    data = torch.load(f"dataset/ogbn-products/processed_data_link_notest.pt")
    x = torch.load(f"dataset/ogbn-products/{emb}_x.pt")
    row, col = data.edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    link_test_path = os.path.join(f"dataset/ogbn-products", "edge_sampled_2_10_only_test.jsonl")
    with open(link_test_path, 'r') as f:
        link_test_lines = f.readlines()
    link_test_lines = [json.loads(line) for line in link_test_lines]
    n = data.num_nodes
    mask = torch.full([n], fill_value=False, dtype=torch.bool)
    for link in link_test_lines:
        mask[link['id'][0]] = True
        mask[link['id'][1]] = True
    mp = MP()
    torch.save(mask, f"dataset/ogbn-products/no_test_link_mask.pt")
    for i in range(4):
        x = mp.partition_propagate(data.edge_index, x=x, norm=norm, chunk_size=200, cuda=True)
        torch.save(x[mask].cpu(), f"dataset/ogbn-products/{emb}_{i + 1}hop_x_notestlink.pt")


def generate_multi_hop_x(dataset, emb="sbert"):
    data = torch.load(f"dataset/{dataset}/processed_data.pt")
    x = torch.load(f"dataset/{dataset}/{emb}_x.pt")
    row, col = data.edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mp = MP()
    for i in range(4):
        x = mp.propagate(data.edge_index, x=x, norm=norm)
        torch.save(x, f"dataset/{dataset}/{emb}_{i+1}hop_x.pt")

def get_sbert_embedding(texts, device):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def build_laplacian_emb(k_hop, sample_size):
    n = int(((sample_size ** (k_hop+1)) -1) / (sample_size - 1))
    edge_row = []
    edge_col = []
    last_hop_start = last_hop_end = 0
    for i in range(k_hop):
        edge_row.extend([x for x in range(last_hop_start, last_hop_end+1) for _ in range(sample_size)])
        edge_col.extend(list(range(last_hop_start*sample_size+1, last_hop_end*sample_size+sample_size+1)))
        last_hop_start = last_hop_start*sample_size+1
        last_hop_end = last_hop_end*sample_size+sample_size
    edge_row = np.array(edge_row)
    edge_col = np.array(edge_col)
    # in_degree=1
    A = sp.coo_matrix((np.array([1]*len(edge_row)),(edge_col, edge_row)), shape=(n,n))
    L = sp.eye(n) - A

    EigVal, EigVec = np.linalg.eig(L.toarray())

    PE = torch.FloatTensor(EigVec)
    # # get random flip signs
    # emb_dim = EigVec.shape[1]
    # rand_sign = 2 * (np.random.rand(emb_dim) > 0.5) - 1.
    # PE = torch.FloatTensor(rand_sign * topk_EigVec)
    torch.save(PE, f"dataset/laplacian_{k_hop}_{sample_size}.pt")
    return PE