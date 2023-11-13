import fire
import json


def create_dataset(
    file_path_in: str,
    file_path_out: str,
):
    """Function to take a JSON array of single turn conversations
    add a sys prompt and format as a dataset for llama_2

    Parameters
    ----------
    file_path_in : str
        Path to JSON file containing list of dicts
        with single turn conversations
    file_path_out : str
        Path to save JSONL dataset to
    """
    with open(file_path_in, "r") as f:
        data = json.load(f)

    # Remove any item where video is in the url, or in the title
    # regardless of case
    # Log how many items removed
    data = [d for d in data if "video" not in d["title"].lower()]
    data = [d for d in data if "video" not in d["url"].lower()]

    dataset = []
    for d in data:
        dataset.append(d)

    # Write out as JSONL file
    with open(file_path_out, "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    fire.Fire()
