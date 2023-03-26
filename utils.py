from typing import Dict, Union
from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import get_scheduler


# Helper functions


def load_json(file_path: Union[Path, str]) -> pd.DataFrame:
    """jsonl_to_df read jsonl file and return a pandas DataFrame.

    Args:
        file_path (Union[Path, str]): The jsonl file path.

    Returns:
        pd.DataFrame: The jsonl file content.

    Example:
        >>> read_jsonl_file("data/train.jsonl")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    with open(file_path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def jsonl_dir_to_df(dir_path: Union[Path, str]) -> pd.DataFrame:
    """jsonl_dir_to_df read jsonl dir and return a pandas DataFrame.

    This function will read all jsonl files in the dir_path and concat them.

    Args:
        dir_path (Union[Path, str]): The jsonl dir path.

    Returns:
        pd.DataFrame: The jsonl dir content.

    Example:
        >>> read_jsonl_dir("data/extracted_dir/")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    print(f"Reading and concatenating jsonl files in {dir_path}")
    return pd.concat(
        [pd.DataFrame(load_json(file)) for file in Path(dir_path).glob("*.jsonl")]
    )


def generate_evidence_to_wiki_pages_mapping(
    wiki_pages: pd.DataFrame,
) -> Dict[str, Dict[int, str]]:
    """generate_wiki_pages_dict generate a mapping from evidence to wiki pages by evidence id.

    Args:
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame:
    """

    def make_dict(x):
        result = {}
        sentences = re.split(r"\n(?=[0-9])", x)
        for sent in sentences:
            splitted = sent.split("\t")
            if len(splitted) < 2:
                # Avoid empty articles
                return result
            result[splitted[0]] = splitted[1]
        return result

    # copy wiki_pages
    wiki_pages = wiki_pages.copy()

    # generate parse mapping
    print("Generate parse mapping")
    wiki_pages["evidence_map"] = wiki_pages["lines"].parallel_map(make_dict)
    # generate id to evidence_map mapping
    print("Transform to id to evidence_map mapping")
    mapping = dict(
        zip(
            wiki_pages["id"].to_list(),
            wiki_pages["evidence_map"].to_list(),
        )
    )
    # release memory
    del wiki_pages
    return mapping


def set_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
):
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def save_checkpoint(model, ckpt_dir: str, current_step: int, mark: str = ""):
    if mark != "":
        mark += "_"
    torch.save(model.state_dict(), f"{ckpt_dir}/{mark}model.{current_step}.pt")


def load_model(model, ckpt_name, ckpt_dir: str):
    model.load_state_dict(torch.load(f"{ckpt_dir}/{ckpt_name}"))
    return model
