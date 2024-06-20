export ASCEND_RT_VISIBLE_DEVICES=2,3

output_dir=test_temporal_block_log

msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9002 --log_dir=$output_dir/parallel_logs opensora/test/test_temporal_block.py
