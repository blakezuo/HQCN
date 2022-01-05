
CUDA_VISIBLE_DEVICES=0 python -W ignore run.py \
--n_layers 1 \
--output_dir /home/xiaochen_zuo/sigir2021/eagle/output_tg/checkpoint-10000 \
--per_gpu_train_batch_size=64 \
--per_gpu_eval_batch_size=50 \
--per_gpu_test_batch_size=50 \
--learning_rate=1e-4 \
--num_train_epochs=20 \
--logging_steps=5 \
--save_steps=2000 \
--dataset=tiangong \
--history_num=5 \
--num_workers=2 \
--fp16=False \
--overwrite_output_dir=True \
--warmup_portion=0.1 \
--do_train=True \
--do_eval=False \
--gradient_accumulation_steps=4
