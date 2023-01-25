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

fig_paths = {}

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
	"changing_omega/figures/ae_time_omega/exp_6.pdf": "syn_ae_time_omega.pdf",
	"changing_omega/figures/cae_time_omega/exp_6.pdf": "syn_cae_time_omega.pdf",
	"interference/figures/cae_time_omega/exp_1.pdf": "syn_interference.pdf",
	"chaos_to_order/figures/cae_time_omega/exp_1.pdf": "syn_chaos_to_order.pdf"
}
for fig_file_org_name, fig_file_new_name in fig_file_names.items():
	fig_paths[os.path.join(fig_dir_path, fig_file_org_name)] = os.path.join(selected_figs_dir_path, fig_file_new_name)

# hasc
fig_dir_path = os.path.join(
	root_dir_path,
	"test",
	"systematic_tests",
	"hasc",
	"main",
	"batch_output"
)
fig_file_names = {
	"fixed_omega/figures/mae_omega/HASC_ALL.pdf": "hasc_mae_omega_all.pdf",
	"changing_omega/figures/cae_time_omega/HASC1001_26_469.pdf": "hasc_cae_sample.pdf"
}
for fig_file_org_name, fig_file_new_name in fig_file_names.items():
	fig_paths[os.path.join(fig_dir_path, fig_file_org_name)] = os.path.join(selected_figs_dir_path, fig_file_new_name)

# arcelik
fig_dir_path = os.path.join(
	root_dir_path,
	"test",
	"systematic_tests",
	"arcelik",
	"batch_output"
)
fig_file_names = {
	"fixed_omega/figures/mae_omega/atlas_mae_omega_all.pdf": "atlas_mae_omega_all.pdf",
	"changing_omega/figures/cae_time_omega/1_363_lrc5_orc76.pdf": "atlas_cae_sample.pdf"
}
for fig_file_org_name, fig_file_new_name in fig_file_names.items():
	fig_paths[os.path.join(fig_dir_path, fig_file_org_name)] = os.path.join(selected_figs_dir_path, fig_file_new_name)

# copy and rename
for fig_file_org_path, fig_file_new_path in fig_paths.items():
	print(fig_file_org_path, fig_file_new_path)
	shutil.copy(fig_file_org_path, fig_file_new_path)