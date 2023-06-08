# EntityResolutionWithPyG
EmbDI is a framework that uses a graph as an intermediate representation for relational tables to generate embeddings of schema
elements such as tuples, tokens, and columns. It was proved that this
approach causes a lower loss of information if compared to frameworks that
consider tuples as sentences.
In this project, I re-implemented from scratch EmbDI using PyTorch geometric to generate the embeddings. My objective was to test the performances
of this deep learning library based on Graph Neural Networks on the entity
resolution task.

[Reference paper: Creating Embeddings of Heterogeneous Relational
Datasets for Data Integration Tasks](https://doi.org/10.1145/3318464.3389742)
