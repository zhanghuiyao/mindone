export ASCEND_RT_VISIBLE_DEVICES=0,1

output_dir=test_mha_log

msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9001 --log_dir=$output_dir/parallel_logs opensora/test/test_mha.py
