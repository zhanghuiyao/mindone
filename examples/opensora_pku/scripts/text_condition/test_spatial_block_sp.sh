export ASCEND_RT_VISIBLE_DEVICES=2,3

output_dir=test_spatial_block_log

msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9002 --log_dir=$output_dir opensora/test/test_spatial_block.py > _log_msrun.txt 2>&1 &
