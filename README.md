# Citation Recommendation System using Attention Mechanisms

Imagine writing a novel scientific paper that has numerous citation requirements. Given the mammoth amount of research that could be present in a particular field, there might be several papers that you are unaware of and could be referenced in the publication. How about an algorithm that could provide you with recommendations of the publications in the field on which you are working? The recommendations could be ordered according to the date of publication with priority of most recent to least recent. Well, this is what we want to achieve in our project. We aim to develop a language model that can summarize the written document and finally, recommend citation of papers that could potentially be included in the ”References” section. This might seem trivial at first thought, but as we delve to segment the task modules, we understand the sophistication involved. We want to experiment on various transformer architectures that could solve this task along with generating position numbers for the citations in accord with the context. The position number generation is something entirely novel that we intend to add up to existing solutions which involve around document summarization and citation recommendation.

We present an extensive research using BERT's classification network for Binary Classification of documents. We also extend this to BERT's language representations for document summarization embeddings using robust similarity indexes. We conducted research on benchmark datasets like [OpenCorpus](http://opus.nlpl.eu/) and [DBLP v10](https://dblp.org/). 

**CUDA SETUP**
```
CUDA: '10.2.89'    
CuDNN: 7603 
```

**CONDA ENVIRONMENT**
```
conda create -n document_citation python=3.7
pip install numpy
pip install pandas
pip install transformers
pip install -U scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

To train and evaluate on the Classification Task, the commands can be passed as 

```
python --task Classification --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

To train and evaluate on the Contrastive Task, the commands can be passed as 

```
python --task Contrastive --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

To train and evaluate on [OpenCorpus](http://opus.nlpl.eu/) and [DBLP v10](https://dblp.org/), the commands can be passed as 

```
python --data_type opencorpus --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

To train and evaluate on [DBLP v10](https://dblp.org/), the commands can be passed as 

```
python --data_type dblp --batch_size 16 --epoch 50 --preprocess --data_path /data/

```

There are also options for incorporating various losses in the Contrastive Task like [Contrastive Loss](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) and [Cosine Embedding Loss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html). They can be accessed via the ```--loss``` argument as

```
python --task Contrastive --loss contrastive/cosine_embedding/margin_ranking --batch_size 16 --epoch 50 --preprocess --data_path /data/

```
To test the model, a pre-trained model weight can be used which should be saved in the **BERT_CLassification_models** or **BERT_Contrastive_models** folder! An example would be like 

```
python main.py --mode test --batch_size 16 --pretrained_model model_folder/model.pth 

```




