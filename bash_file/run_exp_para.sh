#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
NUM_TESTS=20 # Default number of tests for overall coverage

# --- Input Validation ---
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <provider> <model_name>"
  echo "  provider:   openai, gemini, or hf"
  echo "  model_name: The specific model identifier (e.g., gpt-4, gemini-2.0-flash, meta-llama/Llama-2-7b-chat-hf)"
  exit 1
fi

PROVIDER=$1
MODEL_NAME=$2

# --- Provider Check & Script Prefix ---
case "$PROVIDER" in
  openai|gemini|hf)
    # Provider is valid
    ;;
  *)
    echo "Error: Invalid provider '$PROVIDER'. Must be 'openai', 'gemini', or 'hf'."
    exit 1
    ;;
esac

# --- GNU Parallel Check ---
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel is not installed. Please install it (e.g., 'sudo apt install parallel' or 'brew install parallel') or use the basic bash version of this script."
    exit 1
fi

# Sanitize model name for use in directory paths (replace / with _)
MODEL_NAME_SANITIZED=$(echo "$MODEL_NAME" | tr '/' '_')

# --- Directory Setup ---
RUN_DIR="experiment_results/${PROVIDER}_${MODEL_NAME_SANITIZED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Results will be stored in: $RUN_DIR"
echo "Using GNU Parallel to run jobs concurrently."

# --- Helper Function for Running a Pipeline ---
# This function will be exported so 'parallel' can use it.
# Usage: run_pipeline <exp_name> <gen_script> <format_mode> <eval_script> [gen_args...]
run_pipeline() {
  local exp_name=$1
  local gen_script=$2
  local format_mode=$3
  local eval_script=$4
  # GNU Parallel passes arguments differently, need to access globals or pass explicitly
  local provider_global=$PROVIDER
  local model_name_global=$MODEL_NAME
  local run_dir_global=$RUN_DIR
  shift 4 # Remove the first four arguments
  local gen_args=("$@") # Capture remaining arguments like --num_tests

  local exp_dir="$run_dir_global/$exp_name"
  local raw_tests_path="$exp_dir/raw_tests_${provider_global}_${model_name_global//\//_}.json" # Include more detail
  local fmt_tests_path="$exp_dir/formatted_tests.json"
  local eval_results_path="$exp_dir/eval_results.json"

  mkdir -p "$exp_dir"
  echo "Starting experiment: $exp_name (via parallel)"

  # Run the 3 steps sequentially; exit on failure (handled by set -e)
  echo "  [${exp_name}] Generating..."
  python "$gen_script" --model "$model_name_global" --output "$raw_tests_path" "${gen_args[@]}" && \
  echo "  [${exp_name}] Formatting..." && \
  python format.py --mode "$format_mode" --path "$raw_tests_path" --output "$fmt_tests_path" && \
  echo "  [${exp_name}] Evaluating..." && \
  python "$eval_script" --path "$fmt_tests_path" --output "$eval_results_path"

  echo "Finished experiment: $exp_name"
}

# Export the function and necessary variables so 'parallel' can access them
export -f run_pipeline
export PROVIDER MODEL_NAME RUN_DIR NUM_TESTS

# --- Define Jobs for GNU Parallel ---
# Use ::: to provide arguments to the commands run by parallel.
# Each argument set corresponds to one call to run_pipeline.

parallel --halt soon,fail=1 --joblog "$RUN_DIR/parallel_joblog.log" --tag --line-buffer run_pipeline ::: \
  "overall_cov" "generate_cov_${PROVIDER}.py" "overall" "eval_overall.py" "--num_tests" "$NUM_TESTS" \
  "line_cov" "generate_targetcov_${PROVIDER}.py" "line" "eval_linecov.py" "--covmode" "line" \
  $(if [ "$PROVIDER" == "openai" ] || [ "$PROVIDER" == "hf" ]; then echo "\"line_cov_cot\" \"gen_linecov_cot_${PROVIDER}.py\" \"line\" \"eval_linecov.py\""; else echo ""; fi) \
  "branch_cov" "generate_targetcov_${PROVIDER}.py" "branch" "eval_branchcov.py" "--covmode" "branch" \
  "path_cov" "generate_pathcov_${PROVIDER}.py" "overall" "eval_pathcov.py"

# The 'if' condition for CoT is handled by conditionally echoing the argument string.
# If the condition is false, it echoes an empty string which parallel ignores.

echo "All experiments dispatched via GNU Parallel."
echo "Check joblog: $RUN_DIR/parallel_joblog.log"
echo "Results stored in: $RUN_DIR"