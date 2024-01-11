Code for the paper [Investigating semantic subspaces of Transformer sentence embeddings through linear structural probing](https://aclanthology.org/2023.blackboxnlp-1.11/) accepted to BlackboxNLP 2023.

Citation:

```bibtex
@inproceedings{nikolaev-pado-2023-investigating,
    title = "Investigating Semantic Subspaces of Transformer Sentence Embeddings through Linear Structural Probing",
    author = "Nikolaev, Dmitry  and
      Pad{\'o}, Sebastian",
    editor = "Belinkov, Yonatan  and
      Hao, Sophie  and
      Jumelet, Jaap  and
      Kim, Najoung  and
      McCarthy, Arya  and
      Mohebbi, Hosein",
    booktitle = "Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.blackboxnlp-1.11",
    doi = "10.18653/v1/2023.blackboxnlp-1.11",
    pages = "142--154",
    abstract = "The question of what kinds of linguistic information are encoded in different layers of Transformer-based language models is of considerable interest for the NLP community. Existing work, however, has overwhelmingly focused on word-level representations and encoder-only language models with the masked-token training objective. In this paper, we present experiments with semantic structural probing, a method for studying sentence-level representations via finding a subspace of the embedding space that provides suitable task-specific pairwise distances between data-points. We apply our method to language models from different families (encoder-only, decoder-only, encoder-decoder) and of different sizes in the context of two tasks, semantic textual similarity and natural-language inference. We find that model families differ substantially in their performance and layer dynamics, but that the results are largely model-size invariant.",
}
```
