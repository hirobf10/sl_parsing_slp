import argparse
from typing import List, Tuple, Union
from collections import defaultdict
import random

import numpy as np


def load_dataset(path, encoding: str):
    with open(path, encoding=encoding) as f:
        dataset = f.readlines()
    return dataset


class Dataset:
    def __init__(self, dataset: List[str]):
        """
        The format of dataset should be CoNLL.
        Tab-separated columns, sentences separated by space.
        each line contains: ID, Word, Base, POS, POS2, ?, Head, Type.
        """
        self.data = self._format(dataset)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Union[int, slice]) -> List[List[str]]:
        if isinstance(key, slice):
            return [self.data[i] for i in range(*key.indices(len(self.data)))]
        elif isinstance(key, int):
            if key < 0:
                key += len(self.data)
            if key < 0 or key >= len(self.data):
                raise IndexError
            return self.data[key]
        else:
            raise TypeError

    def _format(self, dataset: List[str]) -> List[List[List[str]]]:
        format_dataset = []
        for i in range(len(dataset)):
            if dataset[i] == "\n":
                first = i - int(dataset[i - 1].split("\t")[0])
                format_dataset.append(
                    [line.split("\n")[0].split("\t") for line in dataset[first:i]]
                )
        return format_dataset


class FeatureExtractor:
    def __init__(self, dataset: Dataset, lr=0.01, topn=30, init_func=None):
        self.dataset = dataset
        self.lr = lr
        self.topn = topn
        self.init_func = init_func
        self.features = None
        self.weight = None

    def get_weights(self) -> defaultdict[int]:
        self.features = self._get_features()
        self.weights = self._update_weights()
        return self.weights

    def _count_features(self) -> defaultdict[defaultdict[int]]:
        cnt = defaultdict(lambda: defaultdict(int))
        for batch in self.dataset:
            for i, data_i in enumerate(batch):
                cnt["word"][data_i[1]] += 1
                cnt["pos"][data_i[3]] += 1
                cnt["dep"][data_i[-1]] += 1
                if data_i[-1] == "ROOT":
                    cnt["pos_pos"][f"{data_i[3]}_ROOT"] += 1
                    cnt["word_pos"][f"{data_i[1]}_ROOT"] += 1
                    cnt["dep"][f"{data_i[-1]}_ROOT"] += 1
                else:
                    head_idx = i - int(data_i[0]) + int(data_i[6])
                    head = batch[head_idx]
                    cnt["word_word"][f"{data_i[1]}_{head[1]}"] += 1
                    cnt["pos_pos"][f"{data_i[3]}_{head[3]}"] += 1
                    cnt["word_pos"][f"{data_i[1]}_{head[3]}"] += 1
                    cnt["pos_word"][f"{data_i[3]}_{head[1]}"] += 1
                    cnt["dep_word"][f"{data_i[-1]}_{head[1]}"] += 1
                    cnt["dep_pos"][f"{data_i[-1]}_{head[3]}"] += 1
        return cnt

    def _get_features(self) -> List[str]:
        """
        This function extract features for graph-based parser.
        Features include word form, pos, combination of pos and word form.
        """
        cnt = self._count_features()
        features = []
        for feat_type in cnt.keys():
            if feat_type == "pos_pos":
                for feat, v in cnt[feat_type].items():
                    if v >= 5:
                        features.append(feat)
            else:
                for feat, v in sorted(
                    cnt[feat_type].items(), key=lambda i: i[1], reverse=True
                )[: self.topn]:
                    features.append(feat)
        return features

    def _init_weights(self) -> defaultdict[int]:
        weights = defaultdict(int)
        for feat in self.features:
            if self.init_func:
                weights[feat] = self.init_func()
            else:
                weights[feat] = random.random()
        return weights

    def _update_weights(self) -> defaultdict[int]:
        w = self._init_weights()
        for batch in self.dataset:
            score = defaultdict(int)
            for i, data_i in enumerate(batch):
                for j, data_j in enumerate(batch):
                    if i == j:
                        continue
                    score[j] += w[data_i[1]]
                    score[j] += w[data_i[3]]
                    score[j] += w[data_i[-1]]
                    score[j] += w[f"{data_i[1]}_{data_j[1]}"]
                    score[j] += w[f"{data_i[1]}_{data_j[3]}"]
                    score[j] += w[f"{data_i[3]}_{data_j[1]}"]
                    score[j] += w[f"{data_i[3]}_{data_j[3]}"]
                    score[j] += w[f"{data_i[-1]}_{data_j[1]}"]
                    score[j] += w[f"{data_i[-1]}_{data_j[3]}"]
                score["ROOT"] += w[f"{data_i[1]}_ROOT"]
                score["ROOT"] += w[f"{data_i[3]}_ROOT"]
                score["ROOT"] += w[f"{data_i[-1]}_ROOT"]
                if len(score) == 0:
                    continue
                head_pred = sorted(score.items(), key=lambda i: i[1], reverse=True)[0][
                    0
                ]
                head_true = data_i[6]
                if head_pred == "ROOT":
                    if data_i[-1] != "ROOT":
                        w[data_i[1]] -= self.lr
                        w[data_i[3]] -= self.lr
                        w[f"{data_i[1]}_ROOT"] -= self.lr
                        w[f"{data_i[3]}_ROOT"] -= self.lr
                        w[f"{data_i[-1]}_ROOT"] -= self.lr
                elif head_pred != head_true:
                    data_j = batch[head_pred]
                    w[data_i[1]] -= self.lr
                    w[data_i[3]] -= self.lr
                    w[data_i[-1]] -= self.lr
                    w[f"{data_i[1]}_{data_j[1]}"] -= self.lr
                    w[f"{data_i[1]}_{data_j[3]}"] -= self.lr
                    w[f"{data_i[3]}_{data_j[1]}"] -= self.lr
                    w[f"{data_i[3]}_{data_j[3]}"] -= self.lr
                    w[f"{data_i[-1]}_{data_j[1]}"] -= self.lr
                    w[f"{data_i[-1]}_{data_j[3]}"] -= self.lr
        return w


class MSTParser:
    def __init__(self, extractor):
        self.w = extractor.get_weights()

    def __call__(self, batch: List[List[str]]) -> List[int]:
        return self.parse(batch)

    def parse(self, batch: List[List[str]]) -> List[int]:
        scores = self._score(batch)
        best_edges = self._get_best_edges(scores)
        scores = self._subtract(scores, best_edges)
        best_edges = self._get_best_edges(scores)
        status, cycles = self._involveCycle([edge for edge, score in best_edges])
        if status:
            best_edges = self._remove_cycles(scores, cycles, best_edges)
        return [edge for edge, _ in best_edges[1:]]

    def evaluate(self, dataset: Dataset) -> float:
        UAS_SUM = 0
        for batch in dataset:
            # Parse a sentence
            scores = self._score(batch)
            best_edges = self._get_best_edges(scores)
            scores = self._subtract(scores, best_edges)
            best_edges = self._get_best_edges(scores)
            status, cycles = self._involveCycle([edge for edge, score in best_edges])
            if status:
                best_edges = self._remove_cycles(scores, cycles, best_edges)
            y_test = self._extract_test(batch)
            y_pred = [edge for edge, score in best_edges[1:]]
            UAS_SUM += self._get_UAS(y_test, y_pred)
        return UAS_SUM / len(dataset)

    def _score(self, batch: List[List[str]]) -> np.ndarray:
        N = len(batch)
        scores = np.ones([N + 1, N + 1]) * -1e3
        for i in range(1, N + 1):
            for j in range(N + 1):
                if i == j:
                    continue
                score = 0
                score += self.w[batch[i - 1][1]]  # word
                score += self.w[batch[i - 1][3]]  # pos
                score += self.w[batch[i - 1][-1]]
                if j == 0:
                    score += self.w[f"{batch[i-1][1]}_ROOT"] + 1
                    score += self.w[f"{batch[i-1][3]}_ROOT"] + 1
                    score += self.w[f"{batch[i-1][-1]}_ROOT"] + 1
                else:
                    score += self.w[f"{batch[i-1][1]}_{batch[j-1][1]}"]
                    score += self.w[f"{batch[i-1][1]}_{batch[j-1][3]}"]
                    score += self.w[f"{batch[i-1][3]}_{batch[j-1][1]}"]
                    score += self.w[f"{batch[i-1][3]}_{batch[j-1][3]}"]
                    score += self.w[f"{batch[i-1][-1]}_{batch[j-1][1]}"]
                    score += self.w[f"{batch[i-1][-1]}_{batch[j-1][3]}"]
                scores[i, j] = score
        return scores

    def _get_best_edges(self, scores: np.ndarray) -> List[Tuple[int, float]]:
        #         ignore 0-th row because it would contain scores between ROOT as dependent and words as head
        return [
            (np.argmax(scores[i, :]), np.max(scores[i, :])) if i != 0 else (-1, -1e3)
            for i in range(scores.shape[0])
        ]

    #         best_edges = []
    #         root_child_idx = np.argmax(scores[:, 0])
    #         for i in range(scores.shape[0]):
    #             if i == root_child_idx:
    #                 best_edges.append((0, scores[root_child_idx, 0]))
    #             elif i == 0:
    #                 best_edges.append((-1, -1e3))
    #             else:
    #                 head_idx = np.argmax(scores[i, 1:])
    #                 best_edges.append((head_idx, scores[i, head_idx]))
    #         return best_edges

    def _subtract(
        self, scores: np.ndarray, best_edges: List[Tuple[int, float]]
    ) -> np.ndarray:
        N = scores.shape[0]
        for i in range(N):
            for j in range(N):
                if i == 0 or i == j:
                    continue
                scores[i, j] -= best_edges[i][1]
        return scores

    def _involveCycle(self, edges: List[int]) -> Tuple[bool, List]:
        memory = []
        for dep, head in enumerate(edges):
            dep_ = edges[head]
            if dep == dep_ and (sorted([dep, head]) not in memory):
                memory.append(sorted([dep, head]))
        if memory:
            return (True, memory)
        else:
            return (False, [])

    def _remove_cycles(
        self,
        scores: np.ndarray,
        cycles: List[List[int]],
        best_edges: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        N = scores.shape[0]
        for cycle in cycles:
            scores_ = scores.copy()
            scores_[cycle[0], cycle[1]] = -1e3
            scores_[cycle[1], cycle[0]] = -1e3
            node, j = divmod(np.argmax(scores_[cycle, :]), N)
            if node == 0:
                c_head = cycle[0]
            else:
                c_head = cycle[1]
            best_edges[c_head] = (j, scores[c_head, j])
        return best_edges

    def _extract_test(self, batch: List[List[str]]) -> List[int]:
        return [int(data[6]) for data in batch]

    def _get_UAS(self, y_test: List[int], y_pred: List[int]) -> float:
        assert len(y_test) == len(y_pred)
        match_num = 0
        for test, pred in zip(y_test, y_pred):
            if test == pred:
                match_num += 1
        return match_num / len(y_test)


def main(args):
    random.seed(args.seed)
    np.random.seed(seed=args.seed)

    train_dataset = Dataset(load_dataset(args.train_file_path, encoding=args.encoding))
    test_dataset = Dataset(load_dataset(args.test_file_path, encoding=args.encoding))
    extractor = FeatureExtractor(train_dataset, topn=args.topn, lr=args.lr)
    parser = MSTParser(extractor)
    if args.mode == "eval":
        print("UAC: {:.5f}".format(parser.evaluate(test_dataset)))
    else:
        sentence = random.choice(test_dataset)
        print("Sentence:")
        print(" ".join([token[1] for token in sentence]))
        print("Prediction: ")
        print(parser.parse(sentence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="data/mstparser-en-train.dep")
    parser.add_argument("--test_file_path", default="data/mstparser-en-test.dep")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument("--mode", default="eval", choices=["eval", "pred"])
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
