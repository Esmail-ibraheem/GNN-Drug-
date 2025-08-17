# GNN-Drug-

# What the project is
<img width="400" height="300" alt="download (3)" src="https://github.com/user-attachments/assets/264e20e9-ca69-461b-9a7c-401e32e07185" />

A small prototype for **molecular property prediction with a Graph Neural Network (GNN)**—specifically a **SchNet** model—trained on the **QM9** dataset. The presence of a SchNet training script and a saved checkpoint strongly suggests it predicts a scalar molecular property (commonly the dipole moment μ in QM9) and visualizes predicted-vs-true values in a “parity plot.” ([GitHub][1])

# What’s in the repository

* **`train_qm9_schnet.py`** – training script for a SchNet model on the QM9 dataset (SchNet is a continuous-filter convolutional GNN for molecules). ([GitHub][1])
* **`schnet_qm9.pt`** – a saved model checkpoint you can load for testing/inference. ([GitHub][1])
* **`Testing.ipynb`** – a notebook likely used to load the checkpoint, run inference on a test split, and draw the parity plot. ([GitHub][1])
* **`README.md`** – currently minimal (no detailed instructions yet). ([GitHub][1])

# How it (likely) works end-to-end

1. **Load QM9**: Fetches the QM9 dataset (geometries + targets).
2. **Model = SchNet**: Builds a SchNet with learnable atom embeddings and continuous filter blocks.
3. **Train**: Optimizes (typically MSE/MAE loss) to predict a scalar property from each molecular graph.
4. **Save**: Writes weights to `schnet_qm9.pt`.
5. **Evaluate/Plot**: Notebook computes predictions on held-out molecules and makes a parity plot.

(Those steps follow standard SchNet-on-QM9 workflows and match the filenames/checkpoint present in your repo. ([GitHub][1]))

# How to run it (suggested)

Because the README is sparse, here’s a practical path that usually works for SchNet/QM9 projects:

1. Create an environment (Python 3.9–3.11) and install:

   * `torch` (matching your CUDA/CPU),
   * **PyTorch Geometric** stack: `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster` (versions that match your PyTorch),
   * plotting libs (`matplotlib`) if you’ll use the notebook.
2. Train:

   ```bash
   python train_qm9_schnet.py
   ```

   This should download QM9 (first run), train, and save `schnet_qm9.pt`.
3. Test/visualize: open `Testing.ipynb`, load the checkpoint, run inference, and generate the parity plot.

  <img width="490" height="290" alt="download (2)" src="https://github.com/user-attachments/assets/45a03208-64ff-4a0a-a39d-5bc8714f83b6" />


<img width="467" height="470" alt="download" src="https://github.com/user-attachments/assets/82de7fea-fab2-4c76-a9a0-5f28ad8a9214" />

<img width="489" height="490" alt="download (1)" src="https://github.com/user-attachments/assets/17585b71-0db2-44a8-96bd-052d36492e77" />

# What it’s good for (and what it’s not yet)

* ✅ **Good for**: a compact, reproducible **baseline** showing how to train a SchNet on QM9 and produce property predictions.
* ❗**Not yet**: a full **drug-discovery pipeline**. QM9 is small molecules with quantum-chemical labels, not pharmacological endpoints (e.g., solubility, toxicity, bioactivity).



  
