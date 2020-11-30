# Citation Recommendation System using Attention Mechanisms

Imagine writing a novel scientific paper that has numerous citation requirements. Given the mammoth amount of research that could be present in a particular field, there might be several papers that you are unaware of and could be referenced in the publication. How about an algorithm that could provide you with recommendations of the publications in the field on which you are working? The recommendations could be ordered according to the date of publication with priority of most recent to least recent. Well, this is what we want to achieve in our project. We aim to develop a language model that can summarize the written document and finally, recommend citation of papers that could potentially be included in the ”References” section. This might seem trivial at first thought, but as we delve to segment the task modules, we understand the sophistication involved. We want to experiment on various transformer architectures that could solve this task along with generating position numbers for the citations in accord with the context. The position number generation is something entirely novel that we intend to add up to existing solutions which involve around document summarization and citation recommendation.

We present an extensive research using BERT's classification network for Binary Classification of documents. We also extend this to BERT's language representations for document summarization embeddings using robust similarity indexes. We conducted research on benchmark datasets like OpenCorpus and DBLP V12. 

To train and evaluate on the Classification Task, the commands can be passed as 

```
python --task Classification --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

To train and evaluate on the Contrastive Task, the commands can be passed as 

```
python --task Contrastive --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

There are also options for incorporating various losses in the Contrastive Task like [Contrastive Loss](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec), [Cosine Embedding Loss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) and [Margin Ranking Loss](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html). They can be accessed via the ```--loss``` argument as

```
python --task Contrastive --loss contrastive/cosine_embedding/margin_ranking --batch_size 16 --epoch 50 --preprocess --data_path /data/

```




