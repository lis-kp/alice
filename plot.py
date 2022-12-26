import argparse
import torch
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
from datasets import Features, Value, ClassLabel, load_dataset, list_datasets
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt


def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        #print(model(**inputs))
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def tokenize(batch):
    return tokenizer(batch["text"],padding=True,truncation=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting the hidden states')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    train_data = args.train_data

    class_names = ["AFTER", "OVERLAP-OR-AFTER", "OVERLAP", "BEFORE-OR-OVERLAP", "BEFORE", "VAGUE"]

    ft = Features({'id': Value('string'), 'label': ClassLabel(names=class_names), 'text': Value('string')})

    dataset = load_dataset("csv", data_files=train_data, sep="\t", names=["id", "label", "text"], features=ft)

    model_ckpt = "mt-dnn-alice/japanese_bert/"
    # ckpt_path = "/Users/lisk/PycharmProjects/alice_git/gdx_version/alice_github/alice/plot/japanese_bert/jap_bert.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device)
    model_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict, strict=False)

    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    #print(dataset_encoded["train"].column_names)

    dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)

    import numpy as np

    X_train = np.array(dataset_hidden["train"]["hidden_state"])
    y_train = np.array(dataset_hidden["train"]["label"])

    # visualizing the dataset
    from sklearn.preprocessing import MinMaxScaler

    # scale features to [0,1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)

    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)

    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train
    #print(df_emb.head())

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    # fig, axes = plt.subplots(2,2, figsize=(4,4))
    axes = axes.flatten()
    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]

    labels = dataset["train"].features["label"].names

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()
