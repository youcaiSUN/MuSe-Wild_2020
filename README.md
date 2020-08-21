## Multi-modal Continuous Dimensional Emotion Recognition Using Recurrent Neural Network and Self-Attention Mechanism

In this repo, we present our solutions to the MuSe-Wild sub-challenge in [MuSe 2020-The Multimodal Sentiment in Real-life Media Challenge](https://www.muse-challenge.org/).

### Requirements
- Python 3.7
- PyTorch 1.4.0
- Pandas
- Matplotlib
- NumPy
- Pickle

### Model
We utilize the Long Short Term Memory (LSTM) recurrent neural network as well as the
self-attention mechanism (denoted by the dotted lines) for continuous dimensional emotion recognition.

![Architecture](imgs/model.png?raw=true "An overview of our model")


### Fusion
We adopt both early fusion and late fusion for multi-modal emotion
recognition in this challenge. For early fusion, we simply concatenate
multiple uni-modal features and feed them into the model. For late fusion,
we employ a second-level LSTM model to fuse the predictions from
several uni-modal features.

### Usage
1. Change the dataset path in *config.py* to yours.
2. For training the uni-modal and early fusion model, run command like this
```
python main.py --emo_dim_set [arousal or valence] --feature_set [names of your feature sets] ...
```
e.g.,
```
python main.py --emo_dim_set valence --feature_set bert-4 --d_rnn 64 --rnn_n_layers 1 --rnn_bi --attn --n_layers 1 --n_heads 8 --epochs 100 --batch_size 1024 --lr 0.005 --seed 43 --n_seeds 1 --min_lr 1e-5 --rnn_dr 0.0 --attn_dr 0.0 --out_dr 0.0 --win_len 200 --hop_len 100 --add_seg_id --log --gpu 6
```
The above options can be found in *main.py*.
3. For training the late fusion model, run command like this
```
python main_fusion.py --emo_dim_set [arousal or valence] --base_dir [your fusion folder] ...
```
e.g.,
```
python main_fusion.py --emo_dim_set arousal --base_dir ./fusion/test--d_model 32 --rnn_bi --n_layers 1 --epochs 15 --batch_size 64 --lr 0.001 --seed 42 --n_seeds 3 --loss ccc --min_lr 1e-5 --gpu 3 --log
```
The above options can be found in *main_fusion.py*. Note that it's needed to put the multiple uni-modal predictions to the *source* sub-folder in the fusion folder.



### Results
In the sub-challenge, Concordance Correlation Coefficient (CCC) is chosen as the evaluation metric.
The best submission results on validation set and test set are as follows.

| Emotion  | Partition | Baseline [1] | Ours
| ------------- | ------------- | ------------- | ------------- | 
| Arousal  | Val  | 0.3978 | 0.5616 |
| Valence  | Val  | 0.1506 | 0.4876 |
| Arousal  | Test  | 0.2834 | 0.4726 |
| Valence  | Test  | 0.2431 | 0.5996 |


### References

[1] Lukas Stappen, Alice Baird, Georgios Rizos, Panagiotis Tzirakis, Xinchen Du, Felix Hafner, Lea Schumann, Adria Mallol-Ragolta, Bj ̈orn W. Schuller, Iulia Lefter, Erik Cambria, Ioannis Kompatsiaris: “The 2020 Multimodal Sentiment Analysis in Real-life Media Workshop and Challenge: Emotional Car Reviews in-the-wild”, Proceedings of *ACM-MM 2020, Seattle, United States*, 2020.
