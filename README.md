# Sentence-Embedding-Interactive-Analysis

[Embedding Projector](https://projector.tensorflow.org/) is a free web application which offers commonly three methods ("PCA", "t-SNE", and "custom linear projections") for visaulizing high dimensional data. It includes build-in examples for visualizing word embeddings in Natural Language Processing (NLP) and image processing for MNIST in computer vision.

Question may arise in mind, what is PCA, t-SNE and custom linear? 

- [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis): Principal Component Analysis in short for PCA is often effective at exploring the internal structure of the embeddings, revealing the most influential dimensions in the data.
- [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) : t-distributed stochastic neighbor embedding in short for t-SNE is useful for exploring local neighborhoods and finding clusters, allowing developers to make sure that an embedding preserves the meaning in the data
- Custom Linear Projection: Help discover meaningful "directions" in data sets - such as the distinction between a formal and casual tone in a language generation model - which would allow the design of more adaptable ML systems.

To add one more skills in my skills stack, I experimented with a way to load sentence embeddings along with class labels into this tool and explore them interactively. In this repo, I will explain entire process with an example. 

## 1. Preparing Dataset

 To further understand the use case, let's take a subset of 200 movie reviews from the SST-2 dataset that have been classified as positve and negative.
 
 ```
 # import library
import pandas as pd

# import movie review dataset
df = pd.read_csv('http://bit.ly/dataset-sst2', nrows=200, sep='\t', names=['text', 'label'])

# replace target values 1 as positive and 0 as negative
df['label'] = df['label'].replace({1: 'positive', 0:'negative'})
 ```

The dataframe contains the text and label indicating whether it's positive or negative movie reviews. 
```
df.head()
```
![df.head()](https://user-images.githubusercontent.com/40186859/189808973-f0b1820b-32c8-42b8-91ae-fc88c8324da4.png)

Using random text to tamper with five of the responses, we will add noise to our dataset. It will serve as an exception to our example.

**Before Noise**

```
df.loc[[10, 19, 154, 168, 181], 'text']
```

**Output** <br>
![Output before noise](https://user-images.githubusercontent.com/40186859/189809642-82352588-26ee-4a53-b7ae-3a20c5705e83.png)

**After Noise**

```
df.loc[[10, 19, 154, 168, 181], 'text'] = 'asdfg qwerf zxcvb'
df.loc[[10, 19, 154, 168, 181], 'text']
```

**Output** <br>
![Output After Noise](https://user-images.githubusercontent.com/40186859/189809887-77cce82e-7f26-4335-98fc-e124574d394c.png)


## Will Update Soon
