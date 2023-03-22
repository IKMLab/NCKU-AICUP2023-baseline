# %% [markdown]
# # Baseline

# %%
# TODO: type hint for all code cells
# TODO: markdown cells
# TODO: sort imports
# TODO: make some cells hidden by default

# %%
from typing import Dict, List, Tuple
from pathlib import Path
import json
import pickle
import re
import pandas as pd

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)

# %% [markdown]
# Download our starter pack

# %%
!wget #TODO: add link to data
!unzip starterpack.zip

# %% [markdown]
# ## PART 1. Document retrieval

# %% [markdown]
# Prepare the environment and import all library we need

# %%
import hanlp
import opencc
import wikipedia
wikipedia.set_lang("zh")

# %% [markdown]
# Preload the data.

# %%
from utils import load_json

TRAIN_DATA = load_json("data/public_train.jsonl")
TEST_DATA = load_json("data/public_test.jsonl")

# %% [markdown]
# ### Helper function

# %% [markdown]
# 

# %%
def do_st_corrections(text):
    converter_t2s = opencc.OpenCC("t2s.json")
    converter_s2t = opencc.OpenCC("s2t.json")
    simplified = converter_t2s.convert(text)

    return converter_s2t.convert(simplified)

# %% [markdown]
# 

# %%
def get_nps_hanlp(predictor, d) -> List:
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]
    return nps

# %% [markdown]
# 

# %%
def calculate_precision(data: List, predictions: pd.Series):
    precision = 0
    precision_hits = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue
        gt_pages = set(
            [evidence[2] for evidence_set in d["evidence"] for evidence in evidence_set]
        )
        predicted_pages = predictions.iloc[i]
        tmp_precision = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(tmp_precision) / len(predicted_pages)
        precision_hits += 1
        
    print(f"Precision: {precision / precision_hits}")


def calculate_recall(data: List, predictions: pd.Series):
    recall = 0
    recall_hits = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue
        gt_pages = set(
            [evidence[2] for evidence_set in d["evidence"] for evidence in evidence_set]
        )
        predicted_pages = predictions.iloc[i]
        tmp_recall = predicted_pages.intersection(gt_pages)
        recall += len(tmp_recall) / len(gt_pages)
        recall_hits += 1

    print(f"Recall: {recall / recall_hits}")

# %% [markdown]
# 

# %%
def save_aicup_data(
    data: list,
    predictions: pd.Series,
    mode: str = "train",
    num_pred_doc: int = 5,
):
    with open(f"data/{mode}_doc{num_pred_doc}.jsonl", "w") as f:
        for i, d in enumerate(data):
            d["predicted_pages"] = list(predictions.iloc[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# %% [markdown]
# ### Main function for document retrieval

# %%
def get_pred_pages(x):
    results = []
    tmp_muji = []
    mapping = {}  # wiki_page: its index showned in claim
    claim = x["claim"]
    nps = x["hanlp_results"]
    first_wiki_term = []

    for i, np in enumerate(nps):
        wiki_search_results = [do_st_corrections(w) for w in wikipedia.search(np)]
        wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_df = pd.DataFrame(
            {"wiki_set": wiki_set, "wiki_results": wiki_search_results}
        )
        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        muji = grouped_df.index.tolist()

        for prefix, term in zip(muji, candidates):
            if prefix not in tmp_muji:
                matched = False
                if i == 0:
                    first_wiki_term.append(term)
                if (
                    ((new_term := term) in claim)
                    or ((new_term := term.replace("·", "")) in claim)
                    or ((new_term := term.split(" ")[0]) in claim)
                    or ((new_term := term.replace("-", " ")) in claim) # TODO
                ):
                    matched = True

                elif "·" in term:
                    splited = term.split("·")
                    for split in splited:
                        if (new_term := split) in claim:
                            matched = True
                            break

                if matched:
                    # post-processing
                    term = term.replace(" ", "_")
                    term = term.replace("-", "")
                    results.append(term)
                    mapping[term] = claim.find(new_term)
                    tmp_muji.append(new_term)

    if len(results) > 5:
        assert -1 not in mapping.values()
        results = sorted(mapping, key=mapping.get)[:5]
    elif len(results) < 1:
        results = first_wiki_term

    return set(results)

# %% [markdown]
# ### Step 1. Get noun phrases from hanlp consituency parsing tree

# %% [markdown]
# Setup [HanLP](https://github.com/hankcs/HanLP) predictor

# %%
predictor = (
    hanlp.pipeline()
    .append(
        hanlp.load("FINE_ELECTRA_SMALL_ZH"),
        output_key="tok",
    )
    .append(
        hanlp.load("CTB9_CON_ELECTRA_SMALL"),
        output_key="con",
        input_key="tok",
    )
)

# %% [markdown]
# We will skip the process for creating parsing tree when demo on class

# %%
predicted_results = f"data/retrieved_doc.pkl"

if Path(predicted_results).exists():

    with open(predicted_results, "rb") as f:
        hanlp_results = pickle.load(f)

else:

    hanlp_file = f"data/hanlp_con_results.pkl"
    hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA]

    with open(hanlp_file, "wb") as f:
        pickle.dump(hanlp_results, f)

    train_df = pd.DataFrame(TRAIN_DATA)
    train_df.loc[:, "hanlp_results"] = hanlp_results
    predicted_results = train_df.parallel_apply(get_pred_pages, axis=1)

# %%
predicted_results.shape

# %% [markdown]
# ### Step 2. Calculate our results

# %%
calculate_precision(TRAIN_DATA, predicted_results)
calculate_recall(TRAIN_DATA, predicted_results)
save_aicup_data(TRAIN_DATA, predicted_results)

# %% [markdown]
# ### (Optional?) Step 3. Check on our test set

# %%
hanlp_results = [get_nps_hanlp(predictor, d) for d in TEST_DATA]

test_df = pd.DataFrame(TEST_DATA)
test_df.loc[:, "hanlp_results"] = hanlp_results

predicted_results = test_df.parallel_apply(get_pred_pages, axis=1)
save_aicup_data(TEST_DATA, predicted_results, mode="test")

# %% [markdown]
# ## PART 2. Sentence retrieval

# %% [markdown]
# Import some libs

# %%
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)

from utils import (
    load_json,
    jsonl_dir_to_df,
    generate_evidence_to_wiki_pages_mapping,
    set_lr_scheduler,
    save_checkpoint,
    load_model,
)
from dataset import BERTDataset

# %% [markdown]
# Global variable

# %%
SEED = 42

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

DOC_DATA = load_json("data/train_doc5.jsonl")

# GT means Ground Truth
TRAIN_GT, DEV_GT = train_test_split(
    DOC_DATA, test_size=0.2, random_state=SEED, shuffle=True, stratify=ID2LABEL
)

# %% [markdown]
# Preload wiki database

# %%
wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(
    wiki_pages,
)
del wiki_pages

# %% [markdown]
# ### Helper function

# %% [markdown]
# 

# %%
def evidence_macro_precision(
    instance: dict, top_rows: pd.DataFrame
) -> Tuple[float, float]:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [
            [e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None
        ]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] == claim][
            "predicted_evidence"
        ].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (
            this_precision / this_precision_hits
        ) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

# %% [markdown]
# 

# %%
def evidence_macro_recall(
    instance: dict, top_rows: pd.DataFrame
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] == claim][
            "predicted_evidence"
        ].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

# %% [markdown]
# 

# %%
def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    cal_scores: bool = True,
    save_name: str = None,
) -> Dict[str, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        dict[float, float, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = probs
    top_rows = (
        df_evidences.groupby("claim")
        .apply(lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )
    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"data/{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[top_rows["claim"] == claim][
                    "predicted_evidence"
                ].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}

# %% [markdown]
# 

# %%
def get_predicted_probs(model, dataloader) -> np.ndarray:
    """Inference script to get probabilites for the candidate evidence sentences

    Args:
        model: the one from HuggingFace Transformers
        dataloader: devset or testset in torch dataloader

    Returns:
        np.ndarray: probabilites of the candidate evidence sentences
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    return np.array(probs)

# %% [markdown]
# 

# %%
class SentRetrievalBERTDataset(BERTDataset):
    """AicupTopkEvidenceBERTDataset class for AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        sentA = item["claim"]
        sentB = item["text"]

        concat = self.tokenizer(
            sentA,
            sentB,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}
        if "label" in item:
            concat_ten["labels"] = torch.tensor(item["label"])

        return concat_ten

# %% [markdown]
# ### Main function for sentence retrieval

# %%
def pair_with_wiki_sentences(
    mapping: dict,
    df: pd.DataFrame,
    negative_ratio: float,
) -> pd.DataFrame:
    """Only for creating train sentences."""
    claims = []
    sentences = []
    labels = []

    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                page = evidence[2].replace(" ", "_")
                if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                    continue
                sent_idx = str(evidence[3])
                sents.append(mapping[page][sent_idx])

            whole_evidence = " ".join(sents)

            claims.append(claim)
            sentences.append(whole_evidence)
            labels.append(1)

    # negative
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]

        evidence_set = set([(evidence[2], evidence[3])
                            for evidences in df["evidence"][i]
                            for evidence in evidences])
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            # ('城市規劃', sent_idx)
            try:
                page_sent_id_pairs = [(page, sent_idx) for sent_idx in mapping[page].keys()]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                if pair in evidence_set:
                    continue
                text = mapping[page][pair[1]]
                # `np.random.rand(1) <= 0.05`: Control not to add too many negative samples
                if text != "" and np.random.rand(1) <= negative_ratio:
                    claims.append(claim)
                    sentences.append(text)
                    labels.append(0)

    return pd.DataFrame({"claim": claims, "text": sentences, "label": labels})


def pair_with_wiki_sentences_eval(
    mapping: dict,
    df: pd.DataFrame,
    is_testset: bool = False,
) -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []

    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]

        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            # ('城市規劃', sent_idx)
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                text = mapping[page][pair[1]]
                if text != "":
                    claims.append(claim)
                    sentences.append(text)
                    if not is_testset:
                        evidence.append(df["evidence"].iloc[i])
                    predicted_evidence.append([pair[0], int(pair[1])])

    return pd.DataFrame({
        "claim": claims,
        "text": sentences,
        "evidence": evidence if not is_testset else None,
        "predicted_evidence": predicted_evidence,
    })

# %% [markdown]
# ### Step 1. Setup training environment

# %% [markdown]
# Hyperparams

# %%
MODEL_NAME = "bert-base-chinese"
NUM_EPOCHS = 1
LR = 2e-5
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
NEGATIVE_RATIO = 0.03
VALIDATION_STEP = 50
TOP_N = 5

# %% [markdown]
# Experiment Directory

# %%
EXP_DIR = (
    f"sent_retrieval/e{NUM_EPOCHS}_bs{TRAIN_BATCH_SIZE}_"
    + f"{LR}_neg{NEGATIVE_RATIO}_top{TOP_N}"
)
LOG_DIR = "logs/" + EXP_DIR
CKPT_DIR = "checkpoints/" + EXP_DIR

if not Path(LOG_DIR).exists():
    Path(LOG_DIR).mkdir(parents=True)

if not Path(CKPT_DIR).exists():
    Path(CKPT_DIR).mkdir(parents=True)

# %% [markdown]
# ### Step 2. Combine claims and evidences

# %%
train_df = pair_with_wiki_sentences(mapping, pd.DataFrame(TRAIN_GT), NEGATIVE_RATIO)
counts = train_df["label"].value_counts()
print("Now using the following train data with 0 (Negative) and 1 (Positive)")
print(counts)

dev_evidences = pair_with_wiki_sentences_eval(mapping, pd.DataFrame(DEV_GT))

# %% [markdown]
# ### Step 3. Start training

# %% [markdown]
# Dataloader things

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SentRetrievalBERTDataset(train_df, tokenizer=tokenizer)
val_dataset = SentRetrievalBERTDataset(dev_evidences, tokenizer=tokenizer)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE
)
eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)

# %% [markdown]
# Save your memory.

# %%
del train_df

# %% [markdown]
# Trainer

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)

writer = SummaryWriter(LOG_DIR)

# %%
progress_bar = tqdm(range(num_training_steps))
current_steps = 0

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        writer.add_scalar("training_loss", loss.item(), current_steps)

        y_pred = torch.argmax(outputs.logits, dim=1).tolist()

        y_true = batch["labels"].tolist()
        # print(f"batch train acc: {accuracy_score(y_true, y_pred)}")

        current_steps += 1

        if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
            print("Start validation")
            probs = get_predicted_probs(model, eval_dataloader)

            val_results = evaluate_retrieval(
                probs=probs,
                df_evidences=dev_evidences,
                ground_truths=DEV_GT,
                top_n=TOP_N,
            )
            print(val_results)

            # log each metric separately to TensorBoard
            for metric_name, metric_value in val_results.items():
                writer.add_scalar(f"dev_{metric_name}", metric_value, current_steps)

            save_checkpoint(model, CKPT_DIR, current_steps)

print("Finished training!")

# %% [markdown]
# Validation part

# %%
ckpt_name = "model.50.pt"    # You need to change your best checkpoint.
model = load_model(model, ckpt_name, CKPT_DIR)
print("Start final evaluations and write prediction files.")

train_evidences = pair_with_wiki_sentences_eval(wiki_pages, pd.DataFrame(TRAIN_GT))
train_set = SentRetrievalBERTDataset(train_evidences, tokenizer)
train_dataloader = DataLoader(train_set, batch_size=TEST_BATCH_SIZE)

print("Start calculating training scores")
probs = get_predicted_probs(model, train_dataloader)
train_results = evaluate_retrieval(
    probs=probs,
    df_evidences=train_evidences,
    ground_truths=TRAIN_GT,
    top_n=TOP_N,
    save_name=f"train_doc5sent{TOP_N}.jsonl",
)
print(f"Training scores => {train_results}")

print("Start validation")
probs = get_predicted_probs(model, eval_dataloader)
val_results = evaluate_retrieval(
    probs=probs,
    df_evidences=dev_evidences,
    ground_truths=DEV_GT,
    top_n=TOP_N,
    save_name=f"dev_doc5sent{TOP_N}.jsonl",
)
print(f"Validation scores => {val_results}")

# %% [markdown]
# ### (Optional?) Step 4. Check on our test data

# %%
test_data = load_json("data/test_doc5.jsonl")

test_evidences = pair_with_wiki_sentences_eval(wiki_pages, pd.DataFrame(test_data), is_testset=True)
test_set = SentRetrievalBERTDataset(test_evidences, tokenizer)
test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE)

print("Start predicting the test data")
probs = get_predicted_probs(model, test_dataloader)
evaluate_retrieval(
    probs=probs,
    df_evidences=test_evidences,
    ground_truths=test_data,
    top_n=TOP_N,
    cal_scores=False,
    save_name=f"test_doc5sent{TOP_N}.jsonl",
)

# %% [markdown]
# ## PART 3. Claim verification

# %% [markdown]
# import libs

# %%
from typing import Dict, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=4)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from sklearn.metrics import accuracy_score
from utils import (
    load_json,
    jsonl_dir_to_df,
    generate_evidence_to_wiki_pages_mapping,
    set_lr_scheduler,
    save_checkpoint,
    load_model,
)
from dataset import BERTDataset

# %% [markdown]
# Global variables

# %%
SEED = 42

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

TRAIN_DATA = load_json("data/train_doc5sent5.jsonl")
DEV_DATA = load_json("data/dev_doc5sent5.jsonl")

TRAIN_PKL_FILE = Path("data/train_doc5sent5.pkl")
DEV_PKL_FILE = Path("data/dev_doc5sent5.pkl")

# %% [markdown]
# Preload wiki database (same as part 2.)

# %%
wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(
    wiki_pages,
)
del wiki_pages

# %% [markdown]
# ### Helper function

# %% [markdown]
# 

# %%
class AicupTopkEvidenceBERTDataset(BERTDataset):
    """AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        claim = item["claim"]
        evidence = item["evidence_list"]

        # In case there are less than topk evidence sentences
        pad = ["[PAD]"] * (self.topk - len(evidence))
        evidence += pad
        concat_claim_evidence = " [SEP] ".join([*claim, *evidence])

        concat = self.tokenizer(
            concat_claim_evidence,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        label = LABEL2ID[item["label"]] if "label" in item else -1
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}

        if "label" in item:
            concat_ten["labels"] = torch.tensor(label)
        return concat_ten

# %% [markdown]
# 

# %%
def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, device):
    model.eval()

    loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_true.extend(batch["labels"].tolist())

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss.item()
            logits = outputs.logits
            y_pred.extend(torch.argmax(logits, dim=1).tolist())

    acc = accuracy_score(y_true, y_pred)

    return {"val_loss": loss / len(dataloader), "val_acc": acc}

# %% [markdown]
# 

# %%
def run_predict(model: torch.nn.Module, test_dl: DataLoader, device) -> list:
    model.eval()

    preds = []
    for batch in tqdm(test_dl, total=len(test_dl), leave=False, desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch).logits
        pred = torch.argmax(pred, dim=1)
        preds.extend(pred.tolist())
    return preds

# %% [markdown]
# ### Main function

# %%
def join_with_topk_evidence(
    df: pd.DataFrame,
    mapping: dict,
    mode: str = "train",
    topk: int = 5,
) -> pd.DataFrame:
    """join_with_topk_evidence join the dataset with topk evidence.

    Note:
        After extraction, the dataset will be like this:
               id     label         claim                           evidence            evidence_list
        0    4604  supports       高行健...     [[[3393, 3552, 高行健, 0], [...  [高行健 （ ）江西赣州出...
        ..    ...       ...            ...                                ...                     ...
        945  2095  supports       美國總...  [[[1879, 2032, 吉米·卡特, 16], [...  [卸任后 ， 卡特積極參與...
        停各种战争及人質危機的斡旋工作 ， 反对美国小布什政府攻打伊拉克...

        [946 rows x 5 columns]

    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        topk (int, optional): The topk evidence. Defaults to 5.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The dataset with topk evidence_list.
            The `evidence_list` column will be: List[str]
    """

    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns:
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]]
            if not isinstance(x[0], list)
            else [x]
            if not isinstance(x[0][0], list)
            else x
        )

    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].parallel_map(
            lambda x: [
                mapping.get(evi_id, {}).get(str(evi_idx), "")
                for evi_id, evi_idx in x  # for each evidence list
            ][:topk]
            if isinstance(x, list)
            else []
        )
        print(df["evidence_list"][:5])
    else:
        # extract evidence
        df["evidence_list"] = df["evidence"].parallel_map(
            lambda x: [
                " ".join(
                    [  # join evidence
                        mapping.get(evi_id, {}).get(str(evi_idx), "")
                        for _, _, evi_id, evi_idx in evi_list
                    ]
                )
                if isinstance(evi_list, list)
                else ""
                for evi_list in x  # for each evidence list
            ][:1]
            if isinstance(x, list)
            else []
        )

    return df

# %% [markdown]
# ### Step 1. Setup training environment

# %% [markdown]
# Hyperparams

# %%
MODEL_NAME = "bert-base-chinese"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
LR = 7e-5
NUM_EPOCHS = 20
MAX_SEQ_LEN = 256
EVIDENCE_TOPK = 5
VALIDATION_STEP = 25

# %% [markdown]
# Experiment Directory

# %%
OUTPUT_FILENAME = "submission.jsonl"

EXP_DIR = (
    f"claim_verification/e{NUM_EPOCHS}_bs{TRAIN_BATCH_SIZE}_"
    + f"{LR}_top{EVIDENCE_TOPK}"
)
LOG_DIR = "logs/" + EXP_DIR
CKPT_DIR = "checkpoints/" + EXP_DIR

if not Path(LOG_DIR).exists():
    Path(LOG_DIR).mkdir(parents=True)

if not Path(CKPT_DIR).exists():
    Path(CKPT_DIR).mkdir(parents=True)

# %% [markdown]
# ### Step 2. Concat claim and evidences

# %%
if not TRAIN_PKL_FILE.exists():
    train_df = join_with_topk_evidence(
        pd.DataFrame(TRAIN_DATA),
        mapping,
        topk=EVIDENCE_TOPK,
    )
    train_df.to_pickle(TRAIN_PKL_FILE, protocol=4)
else:
    with open(TRAIN_PKL_FILE, "rb") as f:
        train_df = pickle.load(f)

if not DEV_PKL_FILE.exists():
    dev_df = join_with_topk_evidence(
        pd.DataFrame(DEV_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    dev_df.to_pickle(DEV_PKL_FILE, protocol=4)
else:
    with open(DEV_PKL_FILE, "rb") as f:
        dev_df = pickle.load(f)

# %% [markdown]
# ### Step 3. Training

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = AicupTopkEvidenceBERTDataset(
    train_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
val_dataset = AicupTopkEvidenceBERTDataset(
    dev_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=TRAIN_BATCH_SIZE,
)
eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)

writer = SummaryWriter(LOG_DIR)

# %%
progress_bar = tqdm(range(num_training_steps))
current_steps = 0

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        writer.add_scalar("training_loss", loss.item(), current_steps)

        y_pred = torch.argmax(outputs.logits, dim=1).tolist()

        y_true = batch["labels"].tolist()
        # print(f"batch train acc: {accuracy_score(y_true, y_pred)}")

        current_steps += 1

        if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
            print("Start validation")
            val_results = run_evaluation(model, eval_dataloader, device)

            # log each metric separately to TensorBoard
            for metric_name, metric_value in val_results.items():
                print(f"{metric_name}: {metric_value}")
                writer.add_scalar(f"{metric_name}", metric_value, current_steps)

            save_checkpoint(
                model,
                CKPT_DIR,
                current_steps,
                mark=f"val_acc={val_results['val_acc']:.4f}",
            )

print("Finished training!")

# %% [markdown]
# ### (Optional?) Step 4. Make your submission

# %%
TEST_DATA = load_json("data/test_doc5sent5.jsonl")
TEST_PKL_FILE = Path("data/test_doc5sent5.pkl")

if not TEST_PKL_FILE.exists():
    test_df = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df.to_pickle(TEST_PKL_FILE, protocol=4)
else:
    with open(test_pkl_file, "rb") as f:
        test_df = pickle.load(f)

test_dataset = AicupTopkEvidenceBERTDataset(
    test_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

# %% [markdown]
# Prediction

# %%
# TODO: change to your best checkpoint
ckpt_name = "val_acc=0.4208_model.75.pt"
model = load_model(model, ckpt_name, CKPT_DIR)
predicted_label = run_predict(model, test_dataloader, device)

# %% [markdown]
# Write files

# %%
predict_dataset = test_df.copy()
predict_dataset["predicted_label"] = list(map(ID2label.get, predicted_label))
predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
    OUTPUT_FILENAME,
    orient="records",
    lines=True,
    force_ascii=False,
)


