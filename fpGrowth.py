import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from collections import Counter

from scipy.signal import freqs
from tqdm import tqdm

#by clyde
#tired

#config
DATA_DIR = "data"
PAYLOADALL_DIR = os.path.join(DATA_DIR, "PayloadsAllTheThings-4.1")
JSONL_PATH = os.path.join(DATA_DIR, "WEB_APPLICATION_PAYLOADS.jsonl")
MIN_SUPPORT = 0.01 #1% support threshold
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok = True)

def loadJsonlPayload():
    payloads = []
    with open(JSONL_PATH, "r", encoding = "utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                payloads.append(data["payload"])
            except:
                continue
    return payloads

def loadPayloadAll():
    payloads = []
    md_FileCount = 0

    for root, _, files in os.walk(PAYLOADALL_DIR):
        for file in files:
            if file.endswith(".md"):
                md_FileCount += 1
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding = "utf-8") as f:
                        content = f.read()

                    codeBlocks = re.findall(r"```(?:[^\n]*\n)?(.*?)```", content, re.DOTALL)
                    inlineCodes = re.findall(r"`(.*?)`", content)

                    for block in codeBlocks:
                        for line in block.strip().split("\n"):
                            if line.strip():
                                payloads.append(line.strip())
                    payloads.extend(inlineCodes)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")

    print(f"Processed {md_FileCount} markdown files")
    return payloads

def securityTokenizer(payload):
    if payload.startswith(("0x", "\\x", "%")) or "\\u" in payload:
        return [payload]

    tokens = re.split(r"([^a-zA-Z0-9_])", payload)
    tokens = [t for t in tokens if t.strip() and t != " "]

    sql_keywords = ["SELECT", "UNION", "OR", "AND", "FROM", "WHERE",
                    "INSERT", "UPDATE", "DELETE", "DROP", "EXEC",
                    "WAITFOR", "DELAY", "SLEEP", "XP_", "CONVERT"]

    processed = []
    i = 0
    while i < len(tokens):
        merged = False
        for keywords in sql_keywords:
            if tokens[i].upper() == keywords:
                processed.append(keywords)
                merged = True
                break
        if merged:
            i += 1
            continue

        if tokens[i].startswith("on") and tokens[i] in ["onload", "onerror", "onclick"]:
            processed.append(tokens[i])
            i += 1
            continue

        if i < len(tokens) - 1 and tokens[i] in ["<", ">", "=", "!", "'", "\""]:
            processed.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            processed.append(tokens[i])
            i += 1

    return processed

def prepareTransactions():
    print("Loading payload data...")

    jsonlPayloads = loadJsonlPayload()
    payloadAll = loadPayloadAll()
    combinedPayloads = jsonlPayloads + payloadAll

    print(f"Total payloads: {len(combinedPayloads)}")

    #tokenize payloads
    print("Tokenizing payloads...")
    transactions = []
    tokenCounter = Counter()

    for payloads in tqdm(combinedPayloads):
        try:
            tokens = securityTokenizer(payloads)
            transactions.append(tokens)
            tokenCounter.update(tokens)
        except Exception as e:
            print(f"Error tokenizing payload: {payloads[:50]}... - {str(e)}")

    minCount = len(transactions) * 0.001
    filteredTransactions = []
    for tokens in transactions:
        filtered = [t for t in tokens if tokenCounter[t] >= minCount]
        if filtered:
            filteredTransactions.append(filtered)

    print(f"Filtered transactions: {len(filteredTransactions)}/{len(transactions)}")
    return filteredTransactions

def trainFPGrowth(transactions):
    print("Preparing the transaction matrix...")

    transactionEncoder = TransactionEncoder()
    transactionEncoder_array = transactionEncoder.fit(transactions).transform(transactions)
    dataframe = pd.DataFrame(transactionEncoder_array, columns = transactionEncoder.columns_)

    print("Running FP-Growth...")
    frequent_itemsets = fpgrowth(
        dataframe,
        min_support = MIN_SUPPORT,
        use_colnames = True
    )

    #add itemset length
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    #sort by support and length
    return frequent_itemsets.sort_values(
        by = ["support", "length"],
        ascending = [False, False]
    )

def generateVisualization(frequent_itemsets, transactions):
    #top 20 frequent tokens
    all_tokens = [token for trans in transactions for token in trans]
    token_counts = Counter(all_tokens)
    top_tokens = token_counts.most_common(20)

    plt.figure(figsize=(12, 8))
    tokens, counts = zip(*top_tokens)
    plt.barh(tokens, counts, color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Frequent Tokens")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_tokens.png"))

    #support distribution
    plt.figure(figsize=(10, 6))
    plt.hist(frequent_itemsets["support"], bins=50, alpha=0.7)
    plt.title("Support Distribution of Frequent Itemsets")
    plt.xlabel("Support")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "support_distribution.png"))

    #itemset length distribution
    plt.figure(figsize=(10, 6))
    frequent_itemsets["length"].value_counts().sort_index().plot(kind='bar')
    plt.title("Itemset Length Distribution")
    plt.xlabel("Itemset Length")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "itemset_lengths.png"))

    plt.close('all')

def saveResults(frequent_itemsets, transactions):
    frequent_itemsets.to_csv(
        os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"),
        index = False
    )

    #save training summary
    summary = {
        "num_transactions": len(transactions),
        "min_support": MIN_SUPPORT,
        "num_frequent_itemsets": len(frequent_itemsets),
        "top_10_itemsets": [
            {"itemset": list(itemset), "support": support}
            for itemset, support in frequent_itemsets[["itemsets", "support"]].head(10).values
        ],
        "token_statistics": {
            "unique_tokens": len(set(token for trans in transactions for token in trans)),
            "avg_tokens_per_transaction": np.mean([len(t) for t in transactions])
        }
    }

    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent = 2)

    generateVisualization(frequent_itemsets, transactions)

def main():
    print("Starting FP-Growth model training...")
    transactions = prepareTransactions()
    frequent_itemsets = trainFPGrowth(transactions)
    saveResults(frequent_itemsets, transactions)
    print("Training completed. Results saved to 'results' directory")

if __name__ == "__main__":
    main()