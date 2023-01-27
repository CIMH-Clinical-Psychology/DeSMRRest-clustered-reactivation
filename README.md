# DeSMRRest-clustered-reactivation

Code for the publication "Clustered Reactivation During Retrieval"

This analysis was run with Python 3.9, but any version >3.8 should work.

### 1. Getting started

First install the requirements using pip `pip install -r requirements.txt`. It is recommended to run this in a dedicated environment not to mix up your current Python installation. You can do so e.g. using [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

```bash
conda create --name desmrrest python=3.9
conda activate desmrrest
# assuming you are in the folder of the repository
pip install -r requirements.txt
```

### 2. Download and setup

Then you need to specify your settings. Open `settings.py` and around line 117 insert where you want to store the data, or (if you have already downloaded it), where the data was saved. You can leave the other parameters the same.

```python
data_dir = '/path/to/data/'           # directory containing the FIF files
cache_dir = f'{data_dir}/cache/'      # used for caching
plot_dir = f'{data_dir}/plots/'       # plots will be stored here
log_dir = f'{data_dir}/plots/logs/'   # log files will be created here
```

Download the experiment files from [Zenodo](https://zenodo.org/record/8001755) into a common folder. Instead of downloading them individually, you they can be downloaded automatically by running `python download_dataset.py` . This will utilize the Python API `pyzenodo3` and download the 100 GB dataset into your `data_dir`. This can take a while. 

### 3. Run analysis

Now you can simply run `run_analysis.py` and after that `run_supplement.py`. I personally used [Spyder](https://spyder-ide.org/ ) to run the script, which also nicely annotates the cells. It's included in Anaconda, so you might already have it installed.
