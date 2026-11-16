# Machine Learning Practical Kit

## Included files
- `ML_Practical_Code_Only.ipynb` — runnable Jupyter notebook with code and short section headings only.
- `diabetes.csv` — synthetic Pima-style dataset for KNN practice (768 rows).
- `sales_data_sample.csv` — synthetic sales-order dataset for K-Means and hierarchical clustering (900 rows).

**Dataset note:** Both CSVs are synthetic educational data generated for this practical. They are not real patient records or a real company's sales records.

## Install required packages
Open Anaconda Prompt / terminal and run:

```bash
pip install pandas numpy matplotlib scikit-learn scipy jupyterlab
```

## Steps to run
1. Extract this ZIP file.
2. Open a terminal in the extracted folder.
3. Run `jupyter lab`.
4. In JupyterLab, open `ML_Practical_Code_Only.ipynb`.
5. Confirm both CSV files are in the same folder as the notebook.
6. Select **Kernel → Restart Kernel and Run All Cells**.
7. For K-Means, inspect the Elbow graph and update `optimal_k` if needed, then rerun the K-Means and later cells.
8. The notebook saves `sales_kmeans_clustered.csv` and `sales_hierarchical_clustered.csv` in the same folder.

## Run in Google Colab
Upload the notebook and both CSV files to Colab. In the left Files panel, make sure the CSV files are in `/content/`, or adjust the `read_csv` paths. Run the cells from top to bottom.
