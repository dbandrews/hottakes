# hottakes


## Setup

```bash
conda create -n hottakes python=3.11 -y
conda activate hottakes
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name hottakes
```


## Scraping

URLs, article details (text, and title), and all comments per article can be collected with:

```
python hottakes/scrape.py
```

This will create the following files:

- `article_urls.csv`
- `article_details.json`
- `article_comments.csv`

## Dataset Creation

Create the SFT dataset, consisting of articles, and upvote/downvote numbers for the top comment on the article with:

```
python hottakes/process_scraped_data.py build_sft_dataset \
    --df_comments_path data/article_comments.csv \
    --df_article_details_path data/article_details.json \
    --output_path data/processed/sft_dataset.jsonl
```

Create the DPO dataset, which consists of comment pairs for each article, currently the top comment, and the worst comment by (downvotes - upvotes) with:

```
python hottakes/process_scraped_data.py build_dpo_dataset \
    --df_comments_path data/article_comments.csv \
    --df_article_details_path data/article_details.json \
    --output_path data/processed/dpo_dataset.jsonl

```

# SFT:

```
python hottakes/sft_trainer.py \ 
--model_name mistralai/Mistral-7B-v0.1 \
--dataset_name data/processed/sft_dataset.jsonl \
--load_in_4bit \
--use_peft \
--batch_size 2 --mlflow_experiment_name=hottakes --mlflow_run_name=mistral-v2 --article_word_threshold=300 \
--mlflow_tracking_uri=http://192.168.0.26:5000 --seq_length=800 --gradient_accumulation_steps=6 \
--pcu_value_threshold=10 --pcd_pcu_ratio_threshold=0.1
```


# DPO:


High grade the dataset to remove low quality comments:

```
python hottakes/dpo_trainer.py \                           
--model_name output/mistral-v2-2020fbf3-bfae-451d-82fb-792ecb2cf0f7/checkpoint-4700 \
--dataset_name data/processed/dpo_dataset.jsonl \        
--load_in_4bit \                                  
--use_peft \
--per_device_train_batch_size 1 --mlflow_experiment_name=hottakes_dpo --mlflow_run_name=mistral-v3 \
--mlflow_tracking_uri=http://192.168.0.26:5000 --gradient_accumulation_steps=128 --learning_rate=1e-4 --max_length=500 --max_prompt_length=300 --beta=0.1 --save_steps=5 --eval_steps=5 --chosen_pcu_threshold=90 --rejected_pcd_threshold=50
```


```
python hottakes/dpo_trainer.py \
--model_name output/mistral-v2-41834087-8a6c-4bac-9735-489842938712/checkpoint-2700 \
--dataset_name data/processed/dpo_dataset.jsonl \
--load_in_4bit \
--use_peft \
--per_device_train_batch_size=1 --mlflow_experiment_name=hottakes_dpo --mlflow_run_name=mistral-v3 \
--mlflow_tracking_uri=http://192.168.0.26:5000 --gradient_accumulation_steps=128 --learning_rate=1e-4 --max_length=500 --max_prompt_length=300 --beta=0.1 --save_steps=5 --eval_steps=20
```