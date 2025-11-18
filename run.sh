# Timing
start_time=$(date +%s)
echo -e "Processing sequences from $2\n\n"
python3 run.py "$@"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo -e "\n\nTotal elapsed time: $elapsed_time seconds"
