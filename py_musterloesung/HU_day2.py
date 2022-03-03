import pickle
from collections import Counter

import numpy as np
import pandas as pd
import spacy

# tqdm shows progress for df.apply
# install with "conda install tqdm" or "pip install tqdm"
from tqdm import tqdm

tqdm.pandas()


# check whether sent contains "Ukraine" AND "Russland"
# if so find adjectives and nouns
# frequency of sents containing / all sents

# read in df containing spacy doc from pickle file (adapt path)
df = pd.read_pickle("data/reden-bundesregierung.pkl")


def extract_sents(row):
    doc = row["doc_object"]

    # count number of sents
    nsents = len([sent for sent in doc.sents])
    row["nsents"] = nsents

    # find relevant sents
    relevant = [
        sent
        for sent in doc.sents
        if "Ukraine" in sent.text
        and any(i in sent.text for i in ["Russland", "Rußland"])
    ]

    nsents_relevant = len(relevant)
    row["relevant_sents"] = relevant
    row["nsents_relevant"] = nsents_relevant

    return row


df = df.progress_apply(lambda row: extract_sents(row), axis=1)

# calc share of relevant sents in total sents per year
nsents = df.nsents.sum()
nsents_relevant = df.nsents_relevant.sum()
df["ratio"] = df["nsents_relevant"] / df["nsents"] * 100
df.datum.dt.year
df["year"] = df.datum.dt.year

grouped = df[["year", "ratio"]].groupby("year").mean()

# plot
fig = plt.figure()
plt.bar(grouped.index, grouped.ratio)
plt.title("Ratio of relevant sents by year")
plt.savefig("ratio.png")


# ADJECTIVES
def get_adjectives(row):
    sents = row["relevant_sents"]
    adjs = list()
    for sent in sents:
        for tok in sent:
            if tok.pos_ == "ADJ":
                adjs.append(tok.text)
    row["adjs"] = adjs
    return row


df = df.progress_apply(lambda row: get_adjectives(row), axis=1)

# flatten list
adjectives = list(matplotlib.cbook.flatten(df.adjs.tolist()))


def get_locations(row):
    sents = row["relevant_sents"]
    locations = list()
    for sent in sents:
        for ent in sent.ents:
            if ent.label_ == "LOC":
                locations.append(ent.text)
    row["locations"] = locations
    return row


df = df.progress_apply(lambda row: get_locations(row), axis=1)


locations = list(matplotlib.cbook.flatten(df.locations.tolist()))

# counter creates a dict with frequency of unique strings in list
locations_freq = Counter(locations)

locs = pd.DataFrame.from_dict(locations_freq, orient="index")
locs = locs.sort_values(0)

fig = plt.figure(figsize=(10, 20))
matplotlib.rcParams.update({"font.size": 10})
plt.barh(locs.index, locs[0])
plt.grid(axis="x")
plt.savefig("locations.png")

print(adjectives)
print(locations)
