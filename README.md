# hottakes


## Setup

```bash
conda create -n hottakes python=3.11 -y
conda activate hottakes
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name hottakes
```


# SFT:

```
python hottakes/sft_trainer.py \                                      
--model_name mistralai/Mistral-7B-v0.1 \
--dataset_name article_comment_no_video_dataset_v2.jsonl \
--load_in_8bit \
--use_peft \
--batch_size 2 --mlflow_experiment_name=hottakes --mlflow_run_name=mistral-v2 --article_word_threshold=300 \
--mlflow_tracking_uri=http://192.168.0.26:5000 --seq_length=800 --gradient_accumulation_steps=6
```


# DPO:

```
python hottakes/dpo_trainer.py \
--model_name output/mistral-v2-a5b0f997-2021-4b9d-850c-e5e34785ae91/checkpoint-1200 \
--dataset_name article_preferences.jsonl \
--load_in_4bit \
--use_peft \
--per_device_train_batch_size 1 --mlflow_experiment_name=hottakes_dpo --mlflow_run_name=mistral-v2 \
--mlflow_tracking_uri=http://192.168.0.26:5000 --gradient_accumulation_steps=64 --max_length=500 --max_prompt_length=300 --beta=0.1 --save_steps=5 --eval_steps=5 
```