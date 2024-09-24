


### Generating a Submission File

Suppose you have come up with a clustering for all the log files listed in `data/log_files`.
The following code snippet shows how you can generate a submission file. 

```python
from tqdm import tqdm
from ml_challenge.utils import generate_submission_file, get_log_fnames

fnames = get_log_fnames()
# your code for to produce cluster labels
labels = your_clustering_pipeline()
generate_submission_file(fnames, labels, 'PATH_TO_YOUR_SUBMISSION_FILE.csv')
```
---
### Submission evaluation

As mentioned earlier, this year's challenge fits the `unsupervised learning` framework.
This makes evaluating submissions less straightforward compared to challenges of previous years.

A good `unsupervised learning` pipeline should be:
1. Scalable (handles growth of data gracefully with low compute complexity)
2. Interactive (accepts human feedback to improve its output)
3. Generalizable (handles any input data thrown to it)
4. Explainable (provides insights about the decisions it makes)

We welcome submissions that exhibit these properties. And we will take them into consideration when judging submissions.

That said, we realize it can be difficult to quantitatively measure the above aspects. 
To this end, we provide a noisy clustering labels of some of the log files (provided in `data/log_metadata/mlchallenge_labels.csv`) that participants can make use of to guide their clustering pipeline.

One can consider these noisy labels either in 1) a `supervised learning` framework - building a pipeline whose performance is tuned to match provided labels; or 2) a `human-feedback-in-the-loop` framework - building a pipeline whose performance improves iteratively as these noisy labels are provided in an online fashion.

If you'd like to evaluate your submission on your machine based on the provided noisy labels in `data/log_metadata/mlchallenge_labels.csv`

```python
from ml_challenge.utils import evaluate_submission_file
result_dict = evaluate_submission_file('PATH_TO_YOUR_SUBMISSION_FILE.csv', str(Path(repo_path / "data" / "log_metadata" / "mlchallenge_labels.csv")))
```

`result_dict` will be a dictionary of typical score metrics that are used in assessing the quality of a given clustering wrt the groundtruth clustering. See [this](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) for more details.

Please recall that the clustering provided in `data/log_metadata/mlchallenge_labels.csv` is not the actual clustering but it is a good-enough reference.

We also provide a script to evaluate a batch of submissions that we will use for final evaluations and we are providing it here for completeness (It will be used besides the `scripts/convert_mlchallenge.py` script to convert the submissions into format relevant to our DV engineers)

   ```shell
   python scripts/evaluate_mlchallenge_submissions.py --help
   ```


The two items discussed above are demonstrated in the `notebooks/simple_baseline.ipynb` notebook.
