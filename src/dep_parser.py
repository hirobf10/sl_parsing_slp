import os
import argparse
from typing import Callable, List, Tuple, Union
from collections import defaultdict
import random

import numpy as np

ID_TO_FEATURE = {
    0: "ID",
    1: "Word",
    2: "Base",
    3: "POS",
    4: "POS2",
    5: "?",
    6: "Head",
    7: "DepType",
}
FEATURE_TO_ID = {v: k for k, v in ID_TO_FEATURE.items()}


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
    dataset: Dataset
    feature_types_path: str
    topn: int
    feature_types: List[str]
    features: List[str]

    def __init__(self, dataset: Dataset, feature_types_path: str, topn=30):
        self.dataset = dataset
        self.topn = topn
        self.feature_types = self._load_feature_types(feature_types_path)
        self.features = None

    def _load_feature_types(self, feature_path: str, encoding="utf-8") -> List[str]:
        with open(feature_path, mode="r", encoding=encoding) as f:
            lines = [line for line in f.read().split("\n")]
        feature_types = []
        for line in lines:
            _, f_child, f_head = map(int, line.split(","))
            if f_head == -1:
                feature_types.append(ID_TO_FEATURE[f_child])
            else:
                feature_types.append(
                    f"{ID_TO_FEATURE[f_child]}_{ID_TO_FEATURE[f_head]}"
                )
        return feature_types

    def _count_features(self) -> defaultdict[defaultdict[int]]:
        cnt = defaultdict(lambda: defaultdict(int))
        for sents in self.dataset:
            for i, data_i in enumerate(sents):
                for f_type in self.feature_types:
                    if "_" not in f_type:
                        cnt[f_type][data_i[FEATURE_TO_ID[f_type]]] += 1
                    else:
                        if data_i[-1] == "ROOT":
                            f_child = f_type.split("_")[0]
                            cnt[f"{f_type}"][
                                f"{data_i[FEATURE_TO_ID[f_child]]}_ROOT"
                            ] += 1
                        else:
                            head_idx = i - int(data_i[0]) + int(data_i[6])
                            head = sents[head_idx]
                            f_child, f_head = f_type.split("_")
                            cnt[f_type][
                                f"{data_i[FEATURE_TO_ID[f_child]]}_{head[FEATURE_TO_ID[f_head]]}"
                            ] += 1
        return cnt

    def extract(self):
        """
                This function extract features for graph-based parser.
        <<<<<<< HEAD
                Features include word form, pos, dependency, combination of them between dependent and head
        =======
                Features include word form, pos, and combination of pos and word form.
        >>>>>>> a51cafd (Fix the way to update parameters correctly)
        """
        cnt = self._count_features()
        features = []
        for feat_type in cnt.keys():
            for feat, _ in sorted(
                cnt[feat_type].items(), key=lambda i: i[1], reverse=True
            )[: self.topn]:
                features.append(feat)
        self.features = features

    def save(self, feature_path: str, encoding="utf-8"):
        with open(feature_path, mode="w", encoding=encoding) as f:
            f.write("\n".join(self.features))

    def load(self, feature_path: str, encoding="utf-8"):
        with open(feature_path, mode="r", encoding=encoding) as f:
            self.features = [line for line in f.read().split("\n")]


class MSTParser:
    extractor: FeatureExtractor
    w: defaultdict(int)

    def __init__(self, extractor: FeatureExtractor):
        self.features = extractor.features
        self.feature_types = extractor.feature_types
        self.w = None

    def __call__(self, sents: List[List[str]]) -> List[int]:
        return self.parse(sents)

    def _init_weights(self, init_func: Callable) -> defaultdict[int]:
        weights = defaultdict(int)
        for feat in self.features:
            if init_func:
                weights[feat] = init_func()
            else:
                weights[feat] = random.random()
        return weights

    def _update_weights(self, sents: List[List[str]], y_pred: List[int], lr: float):
        y_test = self._extract_test(sents)
        for i, (test, pred) in enumerate(zip(y_test, y_pred)):
            pred -= 1
            if test != pred:
                for f_type in self.feature_types:
                    if "_" not in f_type:
                        self.w[f"{sents[i][FEATURE_TO_ID[f_type]]}"] -= lr
                    else:
                        if pred == -1:
                            f_child = f_type.split("_")[0]
                            self.w[f"{sents[i][FEATURE_TO_ID[f_child]]}_ROOT"] -= lr
                        else:
                            f_child, f_head = f_type.split("_")
                            self.w[
                                f"{sents[i][FEATURE_TO_ID[f_child]]}_{sents[pred][FEATURE_TO_ID[f_head]]}"
                            ] -= lr

    def parse(self, sents: List[List[str]]) -> List[int]:
        scores = self._score(sents)
        best_edges = self._get_best_edges(scores)
        scores = self._subtract(scores, best_edges)
        best_edges = self._get_best_edges(scores)
        status, cycles = self._involveCycle([edge for edge, _ in best_edges])
        if status:
            best_edges = self._remove_cycles(scores, cycles, best_edges)
        return [edge for edge, _ in best_edges[1:]]

    def train(self, dataset: Dataset, init_func=None, lr=0.1, iter_num=10):
        self.w = self._init_weights(init_func)
        for _ in range(iter_num):
            for sents in dataset:
                y_pred = self.parse(sents)
                self._update_weights(sents, y_pred, lr)

    def evaluate(self, dataset: Dataset) -> float:
        UAS_SUM = 0
        for sents in dataset:
            # Parse a sentence
            y_test = self._extract_test(sents)
            y_pred = self.parse(sents)
            UAS_SUM += self._get_UAS(y_test, y_pred)
        return UAS_SUM / len(dataset)

    def _score(self, sents: List[List[str]]) -> np.ndarray:
        N = len(sents)
        scores = np.ones([N + 1, N + 1]) * -1e3
        for i in range(1, N + 1):
            for j in range(N + 1):
                if i == j:
                    continue
                score = 0
                for f_type in self.feature_types:
                    if "_" not in f_type:
                        score += self.w[sents[i - 1][FEATURE_TO_ID[f_type]]]
                    else:
                        if j == 0:
                            f_child = f_type.split("_")[0]
                            score += self.w[
                                f"{sents[i-1][FEATURE_TO_ID[f_child]]}_ROOT"
                            ]
                        else:
                            f_child, f_head = f_type.split("_")
                            score += self.w[
                                f"{sents[i-1][FEATURE_TO_ID[f_child]]}_{sents[j-1][FEATURE_TO_ID[f_head]]}"
                            ]
                scores[i, j] = score
        return scores

    def _get_best_edges(self, scores: np.ndarray) -> List[Tuple[int, float]]:
        # ignore 0-th row because it would contain scores between ROOT as dependent and words as head
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

    def _extract_test(self, sents: List[List[str]]) -> List[int]:
        return [int(data[6]) for data in sents]

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
    extractor = FeatureExtractor(train_dataset, args.feature_types_path, args.topn)
    extractor.extract()
    extractor.save(args.feature_path)
    # if os.path.isfile(args.feature_path):
    #     extractor.load(args.feature_path)
    # else:
    #     extractor.extract()
    #     extractor.save(args.feature_path)

    parser = MSTParser(extractor)
    parser.train(test_dataset, init_func=None, lr=args.lr, iter_num=args.iter_num)
    if args.mode == "eval":
        print("UAS: {:.5f}".format(parser.evaluate(test_dataset)))
    else:
        sentence = random.choice(test_dataset)
        print("Sentence:")
        print(" ".join([token[1] for token in sentence]))
        print("Prediction: ")
        print(parser.parse(sentence))
        print("Reference: ")
        print(parser._extract_test(sentence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="data/mstparser-en-train.dep")
    parser.add_argument("--test_file_path", default="data/mstparser-en-test.dep")
    parser.add_argument("--feature_types_path", default="data/feature-types.txt")
    parser.add_argument("--feature_path", default="data/feature.txt")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument("--iter-num", type=int, default=10)
    parser.add_argument("--mode", default="eval", choices=["eval", "pred"])
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
