"""Minimal BM25 retriever over the angiography knowledge base (kb/knowledge.jsonl)."""
import json
import re

from rank_bm25 import BM25Okapi


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Retriever:
    def __init__(self, path):
        self.chunks = [json.loads(l) for l in open(path) if l.strip()]
        self.bm25 = BM25Okapi([tokenize(c["text"] + " " + c["topic"]) for c in self.chunks])

    def search(self, query, k=6):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in order]


if __name__ == "__main__":
    r = Retriever("kb/knowledge.jsonl")
    for c in r.search("left right coronary injection laterality LM LAD LCx RCA segments course "
                      "stenosis narrowing location projection report not visible"):
        print(c["id"])
