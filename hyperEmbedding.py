import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from random import shuffle

from dhg import Graph, Hypergraph
from model import HGNNP
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad() #exp
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    
    loss = F.cross_entropy(outs, lbls) #exp
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, evaluator, test=False):
    net.eval()
    #outs = net(X, G,0)
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res
    

@torch.no_grad()
def save_embedding(net,X,G,no_gene):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook   
    net.eval()  
    net.lkc1.register_forward_hook(get_activation('lkc1'))
    outs = net(X,G)
    outs_first = activation['lkc1']
    outs_first = outs_first[0:no_gene]
    #print("outs_first.shape",outs_first.shape)
    torch.save(outs_first,"prembed.pt") #input

def HyperEmbedding(X,lbl,no_gene,Bipartite):
    set_seed(2000)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu") 
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}]) #exp
    
    
    G = Graph(X.shape[0], Bipartite)
    HG = Hypergraph.from_graph(G) #exp 
    HG.add_hyperedges_from_graph_kHop(G, k=1)
    
      
    train_size = int(0.6*X.shape[0])
    val_size = int(0.2*X.shape[0])
    print("train_size,val_size: ",train_size,val_size)
    
    train_mask = []
    for i in range(X.shape[0]):
      train_mask.append(False)
    train_mask = torch.BoolTensor(train_mask)
    for i in range(train_size):
      train_mask[i]=True
    
    val_mask = []
    for i in range(X.shape[0]):
      val_mask.append(False)
    val_mask = torch.BoolTensor(val_mask)
    for i in range(train_size,train_size+val_size):
      val_mask[i]=True
      
    test_mask = []
    for i in range(X.shape[0]):
      test_mask.append(False)
    test_mask = torch.BoolTensor(test_mask)
    for i in range(train_size+val_size,X.shape[0]):
      test_mask[i]=True
      
    # shuffling train,validation & test for appropriate distribution of gene and aux
    ind_list = [i for i in range(X.shape[0])]
    shuffle(ind_list)
    train_mask = train_mask[ind_list]
    print("train_mask: ",train_mask)
    val_mask = val_mask[ind_list]
    print("val_mask: ",val_mask)
    test_mask = test_mask[ind_list]
    print("test_mask: ",test_mask)
    
    net = HGNNP(X.shape[1], X.shape[1], 2) 
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    X, lbl = X.to(device), lbl.to(device) 
    HG = HG.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    path = "hgnnp_best_trained.model"
    for epoch in range(200):
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask, evaluator)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    torch.save(net,path)
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    print("test...")
    net = torch.load(path)
    res = infer(net, X, HG, lbl, test_mask, evaluator, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    
    
    save_embedding(net,X,HG,no_gene)
  
    