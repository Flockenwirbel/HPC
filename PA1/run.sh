#!/bin/bash

set -euo pipefail

run_case() {
	local nodes="$1"
	local ntasks="$2"
	local ntasks_per_node="$3"
	local target_program="$4"
	local num_elements="$5"
	local input_file="$6"
	shift 6

	echo "==== nodes = ${nodes}, ntasks = ${ntasks}, ntasks_per_node = ${ntasks_per_node}, n = ${num_elements} ===="
	srun \
		--nodes="$nodes" \
		--ntasks="$ntasks" \
		--ntasks-per-node="$ntasks_per_node" \
		--cpu-bind=core \
		"$target_program" "$num_elements" "$input_file" "$@"
}

# Evaluator mode: keep strict compatibility with `run.sh <target_program> <num_elements> <input_file> [extra_args...]`
if [ "$#" -ge 3 ]; then
	target_program="$1"
	num_elements="$2"
	input_file="$3"
	shift 3
	extra_args=("$@")

	max_tasks=56
	min_block=500          # intra-node: shared-memory MPI is cheap, be aggressive
	min_block_cross=18000  # cross-node: amortize IB latency, kicks in at ~1M
	ntasks=1

	# First pass: prefer exact divisors → equal-sized blocks → minimum phases & fast-path merge
	for ((p=max_tasks; p>=2; p--)); do
		mb=$(( p > 28 ? min_block_cross : min_block ))
		if [ $((p > 28 && p % 2 != 0)) -eq 1 ]; then continue; fi
		if [ $((num_elements % p)) -eq 0 ] && [ $((num_elements / p)) -ge $mb ]; then
			ntasks="$p"
			break
		fi
	done

	# Second pass: allow non-divisors if no exact divisor was found
	if [ "$ntasks" -eq 1 ]; then
		for ((p=max_tasks; p>=2; p--)); do
			mb=$(( p > 28 ? min_block_cross : min_block ))
			if [ $((p > 28 && p % 2 != 0)) -eq 1 ]; then continue; fi
			if [ $((num_elements / p)) -ge $mb ]; then
				ntasks="$p"
				break
			fi
		done
	fi

	if [ "$ntasks" -gt 28 ]; then
		nodes=2
		ntasks_per_node=$((ntasks / 2))
	else
		nodes=1
		ntasks_per_node="$ntasks"
	fi

	exec srun \
		--nodes="$nodes" \
		--ntasks="$ntasks" \
		--ntasks-per-node="$ntasks_per_node" \
		--cpu-bind=core \
		"$target_program" "$num_elements" "$input_file" "${extra_args[@]}"
fi

# Report benchmark mode (no args): Collect required timing table on 100000000.dat for  1x1, 1x2, 1x4, 1x8, 1x16, 2x16
default_target="./odd_even_sort"
default_n="100000000"
default_input="/home/course/hpc/assignments/2026/data/public/PA1/100000000.dat"

if [ ! -x "$default_target" ]; then
	echo "Error: ${default_target} not found or not executable. Build first with: make odd_even_sort" >&2
	exit 1
fi

run_case 1 1 1 "$default_target" "$default_n" "$default_input"
run_case 1 2 2 "$default_target" "$default_n" "$default_input"
run_case 1 4 4 "$default_target" "$default_n" "$default_input"
run_case 1 8 8 "$default_target" "$default_n" "$default_input"
run_case 1 16 16 "$default_target" "$default_n" "$default_input"
run_case 2 32 16 "$default_target" "$default_n" "$default_input"
