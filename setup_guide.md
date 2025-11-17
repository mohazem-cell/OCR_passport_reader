# Project Setup Guide

1. **Start a Repository on GitHub**
   - Create a new repository on GitHub to host your project.

2. **Set Up Codespace or Clone Repository**
   - Create a Codespace on the `main` branch, OR
   - Clone the repository to your local device.

3. **Add Requirements File**
   - Create a `requirements.txt` file to list project dependencies.

python3 -m venv venv
source venv/bin/activate

4. **Create Virtual Environment**
   - Install `uv`:
     ```
     pip install uv
     ```
   - Create a virtual environment with Python 3.11:
     ```
     uv venv <env-name> --python 3.11
     OR
     python3 -m venv myvenv
     ```

5. **Activate Virtual Environment**
   - Activate the environment:
     - Linux/macOS:
       ```
       source <env-name>/bin/activate
       ```
     - Windows:
       ```
       <env-name>\Scripts\activate
       ```

6. **Install Requirements**
   - Install dependencies from `requirements.txt`:
     ```
     uv pip install -r requirements.txt --link-mode=copy
     ```

7. **Create Directory Structure**
   - Set up project directories:
     ```
     mkdir data notebooks reports models scripts
     ```
     - In windows

      ```
      foreach ($dir in @("data", "notbooks", "reports", "models", "scripts")) {
      mkdir $dir
      }
      ```

8. **Add Raw Data**
   - Place your raw data files in the `data` directory.

9. **Start a Notebook**
   - Create your first Jupyter notebook in the `notebooks` directory.

10. **Organize Workflow in Notebooks**
    - Split your work into separate notebooks:
      - Exploration
      - Deep Analysis
      - Preprocessing and Feature Engineering
      - Modelling and Hyperparameter Tuning

11. **Save Outputs**
    - Store new data versions in the `data` directory.
    - Save reports in the `reports` directory.
    - Save trained models in the `models` directory.

12. **Finalize Work in Scripts**
    - Create the following scripts in the `scripts` directory:
      - `prepare.py`
      - `train.py`
      - `predict.py`
