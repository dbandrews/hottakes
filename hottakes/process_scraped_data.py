import fire
import json

LLAMA2_TEMPLATE = """
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{input_text} [/INST] {desired_text} </s>
"""


def create_llama_2_dataset(
    file_path_in: str,
    file_path_out: str,
    sys_prompt: str = "You're are a hilarious assistant, who tries to make the funniest comment about any text you are provided.",
    input_text_field: str = "text",
    desired_text_field: str = "comment",
):
    """Function to take a JSON array of single turn conversations
    add a sys prompt and format as a dataset for llama_2

    Parameters
    ----------
    file_path_in : str
        Path to JSON file containing list of dicts
        with single turn conversations
    file_path_out : str
        Path to save huggingface dataset to
    sys_prompt : str, optional
        Instruction for the fine tuning, by default "You're are a hilarious assistant, who tries to make the funniest comment about any text you are provided."
    input_text_field : str, optional
        Field within each dict to insert as the input context text, by default "text"
    desired_text_field : str, optional
        Field within each dict to insert as the desired response, by default "comment"
    """
    with open(file_path_in, "r") as f:
        data = json.load(f)

    dataset = []
    for d in data:
        dataset.append(
            {
                "text": LLAMA2_TEMPLATE.format(
                    system_prompt=sys_prompt,
                    input_text=d[input_text_field],
                    desired_text=d[desired_text_field],
                )
            }
        )

    # Write out as JSONL file
    with open(file_path_out, "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    fire.Fire(create_llama_2_dataset)
