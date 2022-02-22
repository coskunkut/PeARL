import os
import sys
import glob
import shutil
from pathlib import Path

# build paths
root_dir_path = os.path.abspath(os.curdir)

# prepare folder
selected_figs_dir_path = os.path.join(
	root_dir_path,
	"test",
	"selected_figs"
)
Path(selected_figs_dir_path).mkdir(parents=True, exist_ok=True)
files = glob.glob("{}/*".format(selected_figs_dir_path))
for f in files:
	os.remove(f)

# synthetic exps
fig_dir_path = os.path.join(
	root_dir_path,
	"test",
	"systematic_tests",
	"multinomial",
	"batch_output"
)
fig_file_names = {
	"fixed_omega_c4/figures/mae_omega/mae_omega.pdf": "syn_mae_omega_c4.pdf",
	"fixed_omega_c8/figures/mae_omega/mae_omega.pdf": "syn_mae_omega_c8.pdf",
	"fixed_omega_c12/figures/mae_omega/mae_omega.pdf": "syn_mae_omega_c12.pdf",
	"fixed_omega_c16/figures/mae_omega/mae_omega.pdf": "syn_mae_omega_c16.pdf",
	"changing_omega/figures/ae_time_omega/exp_6.pdf": "syn_ae_time_omega.pdf",
	"changing_omega/figures/cae_time_omega/exp_6.pdf": "syn_cae_time_omega.pdf"
}

# hasc
# fig_dir_path = os.path.join(
# 	root_dir_path,
# 	"test",
# 	"systematic_tests",
# 	"hasc",
# 	"main",
# 	"batch_output"
# )
# fig_file_names["fixed_omega_c4/figures/mae_omega/mae_omega.pdf"] = "hasc_mae_omega_all.pdf"

# copy and rename
for fig_file_org_name, fig_file_new_name in fig_file_names.items():
	fig_org_path = os.path.join(fig_dir_path, fig_file_org_name)
	fig_new_path = os.path.join(selected_figs_dir_path, fig_file_new_name)
	shutil.copy(fig_org_path, fig_new_path)