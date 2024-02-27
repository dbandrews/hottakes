# %%
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import time
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from hottakes.scraper import get_article_details
from datasets import load_dataset
from random import randrange


# checkpoint_dir = "output/mistral-v2-2020fbf3-bfae-451d-82db-792ecb2cf0f7/checkpoint-4700"
# checkpoint_dir = "output/mistral-v2-41834087-8a6c-4bac-9735-489842938712/checkpoint-2700"
# checkpoint_dir = "output/mistral-v3-dpo-80b91b5c-a472-4478-a481-08b3fd66128b/checkpoint-165"  # spike in eval accuracy
# checkpoint_dir = "output/mistral-v3-dpo-25012257-c547-42f4-b1a0-9edf8174b35a/checkpoint-100"
# checkpoint_dir = (
#     "output/mistral-v3-dpo-97b42f50-9949-493e-8a57-0cd5dcf00e34/checkpoint-340"  # best qualitatively as of 2024-01-01
# )
checkpoint_dir = "output/mistral-v3-dpo-db20c9b7-8db0-4937-a8b6-65ff4aa77ebf/checkpoint-320"
# checkpoint_dir = "output/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c/checkpoint-100"
# checkpoint_dir = "output/mistral-v3-dpo-543c54f8-a17a-46ff-8ed2-f60a2228e229/checkpoint-370"
# checkpoint_dir = "output/mistral-v3-dpo-80b91b5c-a472-4478-a481-08b3fd66128b/checkpoint-220"
# checkpoint_dir = "output/mistral-v3-dpo-1eea94c8-1a63-40b7-a731-0d06f900c5d7/checkpoint-310"
# checkpoint_dir = "output/v1-checkpoint-500"

model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files="data/processed/sft_dataset.jsonl", split="train")


# %% _________________________________________________________________________________
# sample = dataset[randrange(len(dataset))]
model_merged = model.merge_and_unload()


# %%
# ____________________________________________________
# ____________________________________________________
def get_article_dict(url):
    sample = get_article_details(url)
    # Lower case, and underscore the keys
    sample = {k.lower().replace(" ", "_"): v for k, v in sample.items()}
    sample["title_article_text"] = f"{sample['title']} {sample['article_text']}"
    return sample


sample = get_article_dict("https://www.pinkbike.com/news/thibaut-daprela-joins-the-canyon-cllctv-dh-team.html")

# %%
sample = dataset[randrange(len(dataset))]
prompt = f"""### Instruction:
Use the article title and text below, to write the funniest possible comment about this article.

### Input:
{" ".join(sample['title_article_text'].strip().split(' ')[:300])}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_numel = inputs["input_ids"].numel()

sample_settings_grid = [
    (top_p, temperature) for top_p in np.arange(0.9, 1.1, 0.1) for temperature in np.arange(1.0, 1.1, 0.1)
]
# Generate range of top_p values, temperature values and generate for each
# for top_p, temperature in sample_settings_grid:
for top_p, temperature in [(1.0, 1.0)]:
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            # eos_token_id=[
            #     tokenizer.eos_token_id,
            #     # 13,  # \n
            #     27332,
            #     28771,
            #     842,
            #     28705,
            #     918,
            #     774,
            # ],
        )

    generated_response = tokenizer.batch_decode(
        outputs[:, input_numel:].detach().cpu().numpy(), skip_special_tokens=True
    )[0]

    print(f"-----Title + Article Text:\n{sample['title_article_text']}\n")
    print(f"-----Top_p: {top_p}, Temperature: {temperature}")
    print(f"----Generated Response:\n{generated_response}")
    print(f"-----Ground truth:\n{sample['top_comment_text']}")
    print(f"-----URL:\n{sample['url']}")
    print(f"Time: {time.time() - start_time}")
    print("\n\n")

# %%

# %%
# Upload model to huggingface
# import getpass
# from huggingface_hub import HfApi

# api = HfApi()
# # api.create_repo(
# #     repo_id="dbandrews/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c",
# #     token=getpass.getpass("Token:"),
# #     repo_type="model",
# # )
# api.upload_folder(
#     folder_path="/home/drumm/projects/hottakes/models/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c",
#     repo_id="dbandrews/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c",
#     repo_type="model",
#     token=getpass.getpass("Token:"),
# )


# %%


def get_article_dict(url):
    sample = get_article_details(url)
    # Lower case, and underscore the keys
    sample = {k.lower().replace(" ", "_"): v for k, v in sample.items()}
    sample["title_article_text"] = f"{sample['title']} {sample['article_text']}"
    return sample


sample = get_article_dict("https://www.pinkbike.com/news/orange-bikes-resumes-trading-under-owner-ashley-ball.html")


# sample = dataset[randrange(len(dataset))]
prompt = f"""### Instruction:
Use the article title and text below, to write the funniest possible comment about this article.

### Input:
{" ".join(sample['title_article_text'].strip().split(' ')[:300])}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_numel = inputs["input_ids"].numel()

base_checkpoint_dir = "output/mistral-v3-dpo-543c54f8-a17a-46ff-8ed2-f60a2228e229/"

results = []
for checkpoint in tqdm(Path(base_checkpoint_dir).glob("checkpoint-*")):
    inferece_results = {}
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )

    inferece_results["time"] = time.time() - start_time
    generated_response = tokenizer.batch_decode(
        outputs[:, input_numel:].detach().cpu().numpy(), skip_special_tokens=True
    )[0]

    inferece_results["checkpoint"] = str(checkpoint)
    inferece_results["generated_response"] = generated_response
    results.append(inferece_results)

# %%
df_results = (zs
    pd.DataFrame(results)
    .assign(checkpoint=lambda _df: _df.checkpoint.str.extract(r"checkpoint-(\d+)").astype(int))
    .sort_values("checkpoint")
)


# %%
import pandas as pd
from hottakes.scraper import get_comment_votes

pd.DataFrame(
    get_comment_votes("https://www.pinkbike.com/news/photo-epic-finals-red-bull-hardline-tasmania-2024.html")
).query("comment_id.str.contains('3726236')")
