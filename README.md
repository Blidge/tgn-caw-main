# TGN-CAW: using CAW edge embeddings for TGH memory updates

This is the first repo version - I will edit readme later. For now, I leave the TGN readme here.



## Introduction

skip


#### Paper link: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) and [INDUCTIVE REPRESENTATION LEARNING IN TEMPORAL NETWORKS VIA CAUSAL ANONYMOUS WALKS](https://arxiv.org/pdf/2101.05974.pdf)


## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```



### Model Training

UPD: new CAW specific parameters here:
--pos_dim: shape of MLP encoding for each CAW sequence member(dimension of the positional embedding)
--pos_end: way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous walk
--caw_layers: number of steps in CAW walks(when only 1 value passed in caw_neighbors)
--caw_neighbors: a list of neighbor sampling numbers for different hops, when only a single element is input caw_layers will be activated
--caw_use_lstm: Whether to use LSTM on positional encodings(received from CAWs + MLP)
--use_caw: Whether to add CAW features to messages. False results in vanilla TGN

Self-supervised learning using the link prediction task:
```{bash}
# TGN-attn: Supervised learning on the wikipedia dataset
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: Supervised learning on the reddit dataset
python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10
```

Supervised learning on dynamic node classification (this requires a trained model from 
the self-supervised task, by eg. running the commands above):
```{bash}
# TGN-attn: self-supervised learning on the wikipedia dataset
python train_supervised.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: self-supervised learning on the reddit dataset
python train_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10
```

### Baselines

skip


### Ablation Study

skip


#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --use_memory                 Whether to use a memory for the nodes
  --embedding_module           Type of the embedding module
  --message_function           Type of the message function
  --memory_updater             Type of the memory updater
  --aggregator                 Type of the message aggregator
  --memory_update_at_the_end   Whether to update the memory at the end or at the start of the batch
  --message_dim                Dimension of the messages
  --memory_dim                 Dimension of the memory
  --backprop_every             Number of batches to process before performing backpropagation
  --different_new_nodes        Whether to use different unseen nodes for validation and testing
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --randomize_features         Whether to randomize node features
  --dyrep                      Whether to run the model as DyRep
  --pos_dim                    shape of MLP encoding for each CAW sequence member(dimension of the positional embedding)
  --pos_end                    way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous walk
  --caw_layers                 number of steps in CAW walks(when only 1 value passed in caw_neighbors)
  --caw_neighbors              a list of neighbor sampling numbers for different hops, when only a single element is input caw_layers will be activated
  --caw_use_lstm               Whether to use LSTM on positional encodings(received from CAWs + MLP)
  --use_caw                    Whether to add CAW features to messages. False results in vanilla TGN
```

## TODOs 

Finish the model!

## Cite us

```skip
```


