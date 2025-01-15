# Zomato Restaurants Clustering

- Data cleaning and preprocessing pipeline
- Simple data exploration
- Text embedding on list of cuisines, etc
- Clustering based on restaurant characteristics
- ~~Recommendation system~~ (not done)

## Setup

1. Create a Conda environment: `conda create --name zomato python=3.12`
2. Check if the environment is listed: `conda info --envs`
3. Activate the environment: `conda activate zomato`
4. To automate Conda activation on VS Code:
    - Press Ctrl + Shift + P
    - Python: Select Interpreter
    - Select the `zomato` environment
5. Now everytime you open terminal, `zomato` will be activated (until you quit or switch project)
6. Install PyTorch (CUDA 12.4) dependencies: `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124`
7. Install other dependencies: `pip install -r requirements.txt`
8. To clean-up Conda environment: `conda remove --name zomato --all`

## Visualization

<details>
  <summary>Data distribution</summary>

  ![](img/data_distribution.svg)

</details>

<details>
  <summary>Cluster result</summary>

  ![](img/cluster_result.svg)

</details>

<details>
  <summary>2D PCA vs real restaurant coordinates</summary>

  ![](img/pca_vs_real_coordinate.svg)

</details>

## Todo

- ~~Vocabulary builder and word to index using Polars~~
- Implement `inverse_transform` for word to index
- ~~Test pickling on Polars dataframe and functions~~
- See if I can make clusters less centered around price range
- Map exploration using PyDeck
- Use CuML or migrate preprocessings from Scikit to PyTorch entirely

## Lessons and Recommendation

- Don't set pipeline result back as the original variable. This will make evalution much harder since the original (untransformed) data is not available anymore
- Clustering doesn't guarantee better categorization. Sometimes the results are hard to interpret from human perspective, or the cluster quality is bad no matter how much you tweak it (even with the help of embedding)
  - The data structure must be made with data-driven approach in mind, not as an afterthought, so that human-like labeling (like RFM) can still be done in case of bad cluster result. Example:
    - There should be a primary/dominant cuisine theme even if the restaurant owner can add more cuisines as a list. If a restaurant doesn't have specific cuisine culture (e.g. ice cream), a "General" value can be selected
    - Highlights can be unnested as separate boolean column instead (e.g. debit, credit, reservation, takeaway). This can be done as part of preprocessing but probably less accurate since the list can be anything
  - Since the cluster result is bad (only centered around price range), user based recommendation system may be better in this case, rather than content based. Though I haven't tested further to back my claim

