import torch
import os
import sys
import time
import numpy as np
import pandas as pd

import hashlib
from collections import Counter
import subprocess
import pickle
from linevul.TSParse import TSParse

import pandas as pd
import numpy as np
import json
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from git import Commit, Repo
from pydriller import Repository, ModificationType, Git as PyDrillerGitRepo


def convert_str_to_int(s):
	result = int(''.join(list(str(ord(character)) for character in s))) % 123456
	return result

data_folder = 'data_bigvul'
model_folder = 'linevul/saved_models/checkpoint-best-f1/'
project_name = 'bigvul'
setting = 'my_non_vuln'


def get_cmd(config, model_seed, data_seed, function=False, test_only=False, tp_indices_file=''):
	
	if test_only:
		cmd = f"""python linevul/linevul_main.py \
		  --output_dir=linevul/saved_models \
		  --model_type=roberta \
		  --model_name=model_{config}_{project_name}_{setting}_{data_seed}_all_latest.bin \
		  --tokenizer_name=microsoft/codebert-base \
		  --model_name_or_path=microsoft/codebert-base \
		  --do_test \
		  --write_results=True \
		  --train_data_file={data_folder}/train_{config}_{project_name}_{setting}_{data_seed}_all_latest.parquet \
		  --eval_data_file={data_folder}/val_{project_name}_{setting}_{data_seed}_all.parquet \
		  --test_data_file={data_folder}/test_{project_name}_{setting}_{data_seed}_all.parquet \
		  --epochs 10 \
		  --block_size 512 \
		  --train_batch_size 32 \
		  --eval_batch_size 32 \
		  --learning_rate 1e-5 \
		  --max_grad_norm 1.0 \
		  --evaluate_during_training \
		  --seed {model_seed}  2>&1 | tee linevul/train_logs_bigvul/train_{config}_{project_name}_{setting}_{data_seed}_func_all_latest.log
		"""
		print(cmd)
		return cmd
	
	if function:
		cmd = f"""python linevul/linevul_main.py \
		  --output_dir=linevul/saved_models \
		  --model_type=roberta \
		  --model_name=model_{config}_{project_name}_{setting}_{data_seed}_all_latest.bin \
		  --tokenizer_name=microsoft/codebert-base \
		  --model_name_or_path=microsoft/codebert-base \
		  --do_train \
		  --do_test \
		  --write_results=True \
		  --train_data_file={data_folder}/train_{config}_{project_name}_{setting}_{data_seed}_all_latest.parquet \
		  --eval_data_file={data_folder}/val_{project_name}_{setting}_{data_seed}_all.parquet \
		  --test_data_file={data_folder}/test_{project_name}_{setting}_{data_seed}_all.parquet \
		  --epochs 10 \
		  --block_size 512 \
		  --train_batch_size 32 \
		  --eval_batch_size 32 \
		  --learning_rate 1e-5 \
		  --max_grad_norm 1.0 \
		  --evaluate_during_training \
		  --seed {model_seed}  2>&1 | tee linevul/train_logs_bigvul/train_{config}_{project_name}_{setting}_{data_seed}_func_all_latest.log
		"""
	else:
		cmd = f"""python linevul/linevul_main.py \
		  --output_dir=linevul/saved_models \
		  --model_type=roberta \
		  --model_name=model_{config}_{project_name}_{setting}_{data_seed}_all_latest.bin \
		  --tokenizer_name=microsoft/codebert-base \
		  --model_name_or_path=microsoft/codebert-base \
		  --do_test \
		  --do_local_explanation \
		  --top_k_constant=10 \
		  --do_sorting_by_line_scores \
		  --effort_at_top_k=0.2 \
		  --top_k_recall_by_lines=0.01 \
		  --top_k_recall_by_pred_prob=0.2 \
		  --reasoning_method=attention \
		  --load_results=True \
		  --tp_indices_file={tp_indices_file} \
		  --train_data_file={data_folder}/train_{config}_{project_name}_{setting}_{data_seed}_all_latest.parquet \
		  --eval_data_file={data_folder}/val_{project_name}_{setting}_{data_seed}_all.parquet \
		  --test_data_file={data_folder}/test_{project_name}_{setting}_{data_seed}_all.parquet \
		  --epochs 10 \
		  --block_size 512 \
		  --train_batch_size 32 \
		  --eval_batch_size 32 \
		  --learning_rate 1e-5 \
		  --max_grad_norm 1.0 \
		  --evaluate_during_training \
		  --seed {model_seed}  2>&1 | tee linevul/train_logs_bigvul/train_{config}_{project_name}_{setting}_{data_seed}_line_all_latest.log
		"""
	print(cmd)
	return cmd


start_time = time.time()

n_repeats = 10
metrics = {}
configs = [sys.argv[1]]

for config in configs:
	metrics[config] = {'F1': [], 'Recall': [], 'Precision': [],
					   'Top-10_Accuracy': [], 'IFA': [], 'Effort@0.2Recall': [], 'Recall@0.01LOC': []}
	
for i in range(n_repeats):
	print('#' * 20)
	print(f'Repeat {i + 1}')
	
	print('#' * 20, '\n')
	print('Function-level evaluation starts here!!!')
	
	data_seed = (i + 1) * 10000
	
	for config in configs:
		print('#' * 20, '\n')
		print('Config:', config)
		print('Training the model')
		
		model_seed = convert_str_to_int(f'{config}_{project_name}_{setting}') + (i + 1) * 20000

		with subprocess.Popen(get_cmd(config, model_seed, data_seed, function=True), cwd=None,
						  shell=True, stdout=subprocess.PIPE) as proc:
			output = [x.decode("utf-8") for x in proc.stdout.readlines()]
			for result in output:
				if 'test_recall' in result:
					print(f"Recall: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['Recall'].append(float(result.split(' = ')[1].rstrip()))
				elif 'test_precision' in result:
					print(f"Precision: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['Precision'].append(float(result.split(' = ')[1].rstrip()))
				elif 'test_f1' in result:
					print(f"F1: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['F1'].append(float(result.split(' = ')[1].rstrip()))

		for metric in metrics[config]:
			if len(metrics[config][metric]) > 0:
				print(metric, np.mean(metrics[config][metric]), np.std(metrics[config][metric]))
	
	tp_indices_file = ''
	
	print('#' * 20, '\n')
	print('Line-level evaluation starts here!!!')
	for config in configs:
		print('#' * 20, '\n')
		print('Config:', config)

		model_seed = convert_str_to_int(f'{config}_{project_name}_{setting}') + (i + 1) * 20000

		with subprocess.Popen(get_cmd(config, model_seed, data_seed, function=False, test_only=False, tp_indices_file=tp_indices_file), cwd=None,
						  shell=True, stdout=subprocess.PIPE) as proc:
			output = [x.decode("utf-8") for x in proc.stdout.readlines()]
			for result in output:

				if 'Effort@0.2Recall' in result:
					print(f"Effort@0.2Recall: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['Effort@0.2Recall'].append(float(result.split(' = ')[1].rstrip()))
				elif 'Recall@0.01LOC' in result:
					print(f"Recall@0.01LOC: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['Recall@0.01LOC'].append(float(result.split(' = ')[1].rstrip()))
				elif 'IFA =' in result:
					print(f"IFA: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['IFA'].append(float(result.split(' = ')[1].rstrip()))
				elif 'Top-10_Accuracy' in result:
					print(f"Top-10_Accuracy: {float(result.split(' = ')[1].rstrip())}")
					metrics[config]['Top-10_Accuracy'].append(float(result.split(' = ')[1].rstrip()))

		for metric in metrics[config]:
			if len(metrics[config][metric]) > 0:
				print(metric, np.mean(metrics[config][metric]), np.std(metrics[config][metric]))
		
		print('Removing final/clean model')
		best_model_name = f'model_{config}_{project_name}_{setting}_{data_seed}_all_latest.bin'
		os.remove(f'{model_folder}{best_model_name}')

result_folder = 'results_bigvul/'
result_dict = {'Config': [configs[0]] * n_repeats}

for metric in ['Precision', 'Recall', 'F1', 'Effort@0.2Recall', 'Recall@0.01LOC', 'IFA', 'Top-10_Accuracy']:
	if metric in metrics[configs[0]] and len(metrics[configs[0]][metric]) > 0:
		print(metric, np.mean(metrics[config][metric]), np.std(metrics[config][metric]))
		result_dict[metric] = metrics[configs[0]][metric]
	else:
		result_dict[metric] = [-1] * n_repeats

print('Result dict:', result_dict)
results = pd.DataFrame.from_dict(result_dict)

print(f'Saving results file to {result_folder}{configs[0]}_{project_name}_{setting}_all_latest_10.csv')
results = results[['Config', 'Precision', 'Recall', 'F1', 'Effort@0.2Recall', 'Recall@0.01LOC', 'IFA', 'Top-10_Accuracy']]
print(results.values)
results.to_csv(f'{result_folder}{configs[0]}_{project_name}_{setting}_all_latest_10.csv', index=False)

print('Total execution time:', time.time() - start_time, 's.')
