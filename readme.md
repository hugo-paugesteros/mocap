# Violin Mocap Dataset

A Dataset of Violin Motion Capture takes, along with some Python code to manipulate the data.

## Installation
1. Download the data :

2. Install the Python package
`pip install git+https://github.com/hugo-paugesteros/mocap`

## Usage
```python
# Load the dataset metadata
metadata = pd.read_csv("data/processed/dataset.csv")

# Load one take
take_md = metadata.iloc[0]
take = take_md.mocap.to_take(root_path="dataset/path")

# Get the audio recording of the take
y, sr = take.audio

# Get the mocap data of the take
violin = take.mocap.rigid_body.violin
print(violin.position)
print(violin.rotation)
violin_scroll = take.mocap.rigid_body_marker.violin_scroll
```

## Data organization
|    | violin   | excerpt   |   phase |   take | folder                     |
|---:|:---------|:----------|--------:|-------:|:---------------------------|
|  0 | Klimke   | Mozart    |       1 |      1 | Phase 1/Klimke/Mozart/1    |
|  1 | Klimke   | Mozart    |       1 |      2 | Phase 1/Klimke/Mozart/2    |
|  2 | Klimke   | Glazounov |       1 |      1 | Phase 1/Klimke/Glazounov/1 |
|  3 | Klimke   | Glazounov |       1 |      2 | Phase 1/Klimke/Glazounov/2 |
|  4 | Klimke   | Sibelius  |       1 |      1 | Phase 1/Klimke/Sibelius/1  |
|... | ...      | ...       |     ... |    ... | ...                        |