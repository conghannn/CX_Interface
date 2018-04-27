# Makefile for loading CX data into database.

load:
	python cx_init_db.py -c
	python cx_data_to_json.py
	pyoetl neuropils_cfg.json
	pyoetl neurons_cfg.json
	pyoetl arbors_cfg.json
	python cx_infer_synapses.py
	python cx_create_exec_circ.py

update:
	python cx_assign_params.py

input:
	python cx_gen_input.py -d r2l
	python cx_rearrange_input.py

run:
	srun --gres=gpu:1 python cx_demo.py

vis:
	python visualize_output.py -d r2l

new_vis:
	python new_vis.py -d r2l

link:
	python link.py

run_link_demo:
	srun --gres=gpu:1 python link_demo.py

