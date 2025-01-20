# Zomato Restaurant Recsys

This project uses [Zomato restaurant data](https://www.kaggle.com/datasets/rabhar/zomato-restaurants-in-india) to build a content-based recommendation system. It will show similar restaurants based on cuisines, establishment, etc.

Project stacks:
- Scikit Learn and Polars for preprocessing pipeline
- PyTorch for text embedding
- Plotly for quick EDA
- PyDeck for interactive map
- Qdrant for storing vector & similarity search

Important files:
- `clustering.ipynb`: Model building and EDA
- `recsys.ipynb`: Demo of PyDeck and Qdrant

## Setup

### Conda

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

### Qdrant

1. Install Qdrant (either via Docker or natively)
2. Customize the `Makefile` and `local.yaml` file
3. Run `make qdrant` so that the notebook client can connect to it
4. Open the notebook (`recsys.ipynb`)

## Screenshots

<details>
  <summary>Data Distribution</summary>

  ![](img/data_distribution.svg)

</details>

<details>
  <summary>Cluster Result</summary>

  ![](img/cluster_result.svg)

</details>

<details>
  <summary>2D PCA vs Real Restaurant Coordinates</summary>

  ![](img/pca_vs_real_coordinate.svg)

</details>

<details>
  <summary>Interactive Map Example</summary>

  ![](img/interactive_map.png)

</details>

<details>
  <summary>Similarity Result Response</summary>

  ```python
  [ScoredPoint(id=18730208, version=0, score=0.91323787, payload={'name': 'Jay Guru Bengali Sweets', 'establishment': 'Sweet Shop', 'url': 'https://www.zomato.com/ahmedabad/jay-guru-bengali-sweets-airport-gandhinagar-highway-gandhinagar?utm_source=api_basic_user&utm_medium=api&utm_campaign=v2.1', 'address': 'B-001, Pramukh Arcade - 1, Near Reliance Cross Road Kudasan, Airport Gandhinagar Highway', 'city': 'Gandhinagar', 'locality': 'Airport Gandhinagar Highway', 'latitude': 23.1845279883, 'longitude': 72.62906860560001, 'cuisines': ['Mithai'], 'average_cost_for_two': 100, 'price_range': 1, 'highlights': ['Cash', 'Takeaway Available', 'Indoor Seating', 'Pure Veg', 'Digital Payments Accepted', 'Desserts and Bakes'], 'aggregate_rating': 3.2, 'votes': 24, 'photo_count': 10, 'delivery': 0, 'cluster': 0}, vector=None, shard_key=None, order_value=None),
  ScoredPoint(id=18941127, version=5, score=0.9067461, payload={'name': 'Ram Aur Shyam Golawala', 'establishment': 'Dessert Parlour', 'url': 'https://www.zomato.com/rajkot/ram-aur-shyam-golawala-1-150-feet-ring-road?utm_source=api_basic_user&utm_medium=api&utm_campaign=v2.1', 'address': '1/2, Oscar Complex, Near Indira Circle, 150 Feet Ring Road, Rajkot', 'city': 'Rajkot', 'locality': '150 Feet Ring Road', 'latitude': 22.287809, 'longitude': 70.771714, 'cuisines': ['Desserts', 'Ice Cream'], 'average_cost_for_two': 250, 'price_range': 1, 'highlights': ['Cash', 'Takeaway Available', 'Outdoor Seating', 'Pure Veg', 'Digital Payments Accepted', 'Desserts and Bakes'], 'aggregate_rating': 4.0, 'votes': 130, 'photo_count': 8, 'delivery': 0, 'cluster': 0}, vector=None, shard_key=None, order_value=None),
  ScoredPoint(id=18948982, version=2, score=0.8925876, payload={'name': 'Shere Punjab Ice Cream', 'establishment': 'Dessert Parlour', 'url': 'https://www.zomato.com/kota/shere-punjab-ice-cream-2-chawani?utm_source=api_basic_user&utm_medium=api&utm_campaign=v2.1', 'address': 'Shop 5, In Front Of Shubham Enclave, Bajrang Nagar, Chawani, Kota', 'city': 'Kota', 'locality': 'Chawani', 'latitude': 25.179085064, 'longitude': 75.8607639366, 'cuisines': ['Ice Cream'], 'average_cost_for_two': 100, 'price_range': 1, 'highlights': ['No Seating Available', 'Takeaway Available', 'Cash', 'Pure Veg'], 'aggregate_rating': 3.2, 'votes': 14, 'photo_count': 1, 'delivery': 0, 'cluster': 0}, vector=None, shard_key=None, order_value=None),
  ScoredPoint(id=2601786, version=1, score=0.8874425, payload={'name': 'TOP N TOWN', 'establishment': 'Dessert Parlour', 'url': 'https://www.zomato.com/bhopal/top-n-town-1-arera-colony?utm_source=api_basic_user&utm_medium=api&utm_campaign=v2.1', 'address': 'SHUBHAM, Shop E-4/23, Arera Colony, 10 No. Market,  Bhopal', 'city': 'Bhopal', 'locality': 'Arera Colony', 'latitude': 23.2145196887, 'longitude': 77.4326437718, 'cuisines': ['Desserts', 'Beverages', 'Ice Cream'], 'average_cost_for_two': 200, 'price_range': 1, 'highlights': ['Cash', 'Takeaway Available', 'Delivery', 'Indoor Seating', 'Pure Veg', 'Air Conditioned', 'Desserts and Bakes'], 'aggregate_rating': 3.5, 'votes': 29, 'photo_count': 1, 'delivery': 0, 'cluster': 0}, vector=None, shard_key=None, order_value=None),
  ScoredPoint(id=2301722, version=7, score=0.87815434, payload={'name': 'Satnam Kulfi', 'establishment': 'Dessert Parlour', 'url': 'https://www.zomato.com/kanpur/satnam-kulfi-nandlal-chawraha?utm_source=api_basic_user&utm_medium=api&utm_campaign=v2.1', 'address': '125/52, Lal Quarter, Govind Nagar, Nandlal Chawraha, Kanpur', 'city': 'Kanpur', 'locality': 'Nandlal Chawraha', 'latitude': 26.4498, 'longitude': 80.298636, 'cuisines': ['Desserts', 'Ice Cream'], 'average_cost_for_two': 100, 'price_range': 1, 'highlights': ['Takeaway Available', 'Cash', 'Indoor Seating', 'Desserts and Bakes'], 'aggregate_rating': 4.3, 'votes': 230, 'photo_count': 2, 'delivery': 0, 'cluster': 0}, vector=None, shard_key=None, order_value=None)]
  ```

</details>

## Todo

- ~~Vocabulary builder and word to index using Polars~~
- Implement `inverse_transform` for word to index
- ~~Test pickling on Polars dataframe and functions~~
- Use CuML or migrate preprocessings from Scikit to PyTorch entirely
- See if I can make clusters less centered around price range
- ~~Restaurant exploration using PyDeck map~~
- ~~Restaurant recommendation system using Qdrant~~
- Restaurant recommendation on map click/hover (PyDeck 0.9.1 doesn't support event handler yet)

## Misc

<details>
  <summary>Personal Notes</summary>

- Don't save the pipeline result back to the original variable. This will make evalution much harder since the original (untransformed) data is not available anymore
- Clustering doesn't guarantee better categorization. Sometimes the results are hard to interpret from human perspective, or the cluster quality is bad no matter how much you tweak it (even with the help of embedding)
  - The data structure must be made with data-driven approach in mind, not as an afterthought, so that human-like labeling (like RFM) can still be done in case of bad cluster result. Example:
    - There should be a primary/dominant cuisine theme even if the restaurant owner can add more cuisines as a list. If a restaurant doesn't have specific cuisine culture (e.g. ice cream), a default value can be selected
    - Highlights can be unnested as separate boolean column instead (e.g. debit, credit, reservation, takeaway). This can be done as part of preprocessing but probably less accurate since the text can be anything
- User based recommendation system may be better in case we need to predict next restaurant to go/order, as opposed of content-based which will only show similars restaurants (e.g. why would I go to KFC if I already ate at McDonalds?). Unfortunately, there's no user data to test on this dataset

</details>
