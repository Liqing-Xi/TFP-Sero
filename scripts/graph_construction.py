import os
import pandas as pd
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.spatial import KDTree
from torch.cuda.amp import autocast


MODEL_PATH = "./esm2_t33_650M_UR50D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
model = EsmForMaskedLM.from_pretrained(MODEL_PATH).to(device)
model.eval() 

def get_esm_embeddings(sequence):
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        with autocast():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].squeeze(0)
        embeddings = embeddings[1:-1]
    return embeddings.cpu().numpy()

def parse_pdb(pdb_file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    
    residues = []
    for model_ in structure:
        for chain in model_:
            for residue in chain.get_residues():
                if residue.has_id('CA'):
                    ca = residue['CA']
                    residues.append({
                        'residue_id': residue.get_id()[1],
                        'chain_id': chain.get_id(),
                        'coords': ca.get_coord()
                    })
    return residues

def build_graph(embeddings, pdb_residues):
    G = nx.Graph()

    for idx, (embed, res) in enumerate(zip(embeddings, pdb_residues)):
        G.add_node(idx, feature=embed, chain=res['chain_id'], residue_id=res['residue_id'])

    coords = np.array([res['coords'] for res in pdb_residues])
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=10.0)
    for i, j in pairs:
        distance = np.linalg.norm(coords[i] - coords[j])
        G.add_edge(i, j, weight=distance)
    
    return G

def process_protein(seq_name, sequence, label):
    embeddings = get_esm_embeddings(sequence)
    pdb_path = f'/root/data/train_dataset_250_pdb_files/{seq_name}.pdb'
    pdb_residues = parse_pdb(pdb_path)

    if len(embeddings) != len(pdb_residues):
        print(f"⚠️ Warning: Mismatch in length for {seq_name}: embeddings({len(embeddings)}), pdb({len(pdb_residues)})")
        return
    
    G = build_graph(embeddings, pdb_residues)

    node_features = [G.nodes[node]['feature'] for node in G.nodes()]
    x = torch.FloatTensor(np.array(node_features))

    edge_index = []
    edge_attr = []
    for u, v in G.edges():
        dist = G[u][v]['weight']
        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_attr.append([dist])
        edge_attr.append([dist])
    edge_index = torch.LongTensor(np.array(edge_index).T)
    edge_attr = torch.FloatTensor(np.array(edge_attr))

    y = torch.LongTensor([label])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    pt_output = f'/root/data/graphs_10/{seq_name}.pt'
    torch.save(data, pt_output)
    print(f"✅ Saved PyTorch Data to {pt_output}")

def main():
    print(f"Using device: {device}")
    df = pd.read_csv('/root/data/dataset_250.csv')
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing proteins"):
        seq_name = row['seq_name']
        sequence = row['sequence']
        label = row['label']
        process_protein(seq_name, sequence, label)

if __name__ == "__main__":
    main()
