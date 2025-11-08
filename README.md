# IDC409 Project - Shape Identification

By [Rishi Vora](https://github.com/Vortriz) and [Sparsha Ray](https://github.com/SparshaRay).

---

There are four main files.

 - `src/classic.py` contains the Feature Extraction + Random Forests implementation. The dataset is located at `src/dataset_classic.tar.gz`.
 - `src/cnn.py` contains a convolutional neural network-based approach using tensorflow. The dataset is located at `src/dataset_cnn.tar.gz`.

Both datasets are same, just structured differently for the two approaches.

## Viewing the Notebooks

You can view the pre-run notebooks by simply opening [classic.html](https://vortriz.github.io/idc409-project/src/__marimo__/classic.html) and [cnn.html](https://vortriz.github.io/idc409-project/src/__marimo__/cnn.html) in your browser.

## Instructions

Clone the repository:

```
git clone https://github.com/Vortriz/idc409-project
```

### Dependencies

Install [`uv`](https://docs.astral.sh/uv/#installation).

After that, install all python dependencies:

```
uv sync
```

### Running the codes

#### With marimo (recommended)

```
uv run marimo run src/classic.py
uv run marimo run src/cnn.py
```

#### Without marimo

We have also provided Jupyter notebooks for each marimo notebooks in `src/__marimo__/`, but we don't guarantee that they will run.
