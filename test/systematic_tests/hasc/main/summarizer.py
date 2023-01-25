# batch experiment table generator

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# get experiment id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# get a list of files in experiment folder
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "main")
batch_output_folder_path = os.path.join(test_folder_path, "batch_output", batch_id)
if not os.path.exists(batch_output_folder_path):
	print("Batch output folder not found.")
	sys.exit()
run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")
result_table_path = os.path.join(batch_output_folder_path, "result_table_hasc.csv")

# read csv to a pandas dataframe
raw_table = pd.read_csv(run_logs_path, sep=",", index_col=False)

algorithms_to_run = raw_table.columns[raw_table.columns.str.endswith("_mae")]
algorithms_to_run = [alg[:-4] for alg in algorithms_to_run]

# prepare summary table
summary_table_columns = [
	"sequence_id",
	"alphabet_cardinality"
]
group_by_filter = [
	"sequence_id"
]
if "slwe_e" in algorithms_to_run:
	summary_table_columns.append("SLWE (E)")
	group_by_filter.append("slwe_e_lambda")
if "slwe_t" in algorithms_to_run:
	summary_table_columns.append("SLWE (T)")
	group_by_filter.append("slwe_t_lambda")
if "slwe_p_a" in algorithms_to_run:
	summary_table_columns.append("SLWE-PA")
	group_by_filter.extend([
		"slwe_p_a_eta_min",
		"slwe_p_a_eta_max",
		"slwe_p_a_k",
		"slwe_p_a_L"
	])
if "slwe_p_e" in algorithms_to_run:
	summary_table_columns.append("SLWE-PE")
	group_by_filter.extend([
		"slwe_p_e_eta_min",
		"slwe_p_e_eta_max",
		"slwe_p_e_k",
		"slwe_p_e_L"
	])

summary_table = pd.DataFrame(columns = summary_table_columns)
grouped_table = raw_table.groupby(by=group_by_filter).mean().reset_index()

summary_table["sequence_id"] = grouped_table["sequence_id"]
summary_table["alphabet_cardinality"] = grouped_table["alp_car"]

if "slwe_e" in algorithms_to_run:
	summary_table["SLWE (E)"] = grouped_table["slwe_e_mae"]
if "slwe_t" in algorithms_to_run:
	summary_table["SLWE (T)"] = grouped_table["slwe_t_mae"]
if "slwe_p_a" in algorithms_to_run:
	summary_table["SLWE-PA"] = grouped_table["slwe_p_a_mae"]
if "slwe_p_e" in algorithms_to_run:
	summary_table["SLWE-PE"] = grouped_table["slwe_p_e_mae"]

# summary_table = summary_table.transpose().reset_index()
summary_table.loc["Avg."] = summary_table.mean(numeric_only=True)
summary_table.at["Avg.", "sequence_id"] = "Avg."

# save as csv
summary_table.to_csv(path_or_buf=result_table_path, index=False, header=True, sep=",", float_format="%.3f")

# show message
print("summarize : table generated")
