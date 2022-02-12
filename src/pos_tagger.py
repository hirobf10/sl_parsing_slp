import argparse
from typing import List, Optional, Tuple
from collections import defaultdict, deque
import random


def load_dataset(path: str, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        dataset = [line.splitlines()[0] for line in f.readlines()]
    return dataset


class HMMTagger:
    bos_token: str
    eos_token: str

    def __init__(self, bos_token="<s>", eos_token="</s>"):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.prob_transition = None
        self.prob_emission = None
        self.pos = None

    def _get_count_dict(
        self, dataset: List[str], split_token="_"
    ) -> Tuple[defaultdict[defaultdict[int]], defaultdict[defaultdict[int]]]:
        cnt_transition = defaultdict(lambda: defaultdict(int))
        cnt_emission = defaultdict(lambda: defaultdict(int))

        for line in dataset:
            token_pos = deque(line.split())
            token_pos.appendleft(f"{self.bos_token}{split_token}{self.bos_token}")
            token_pos.append(f"{self.eos_token}{split_token}{self.eos_token}")
            token_pos = list(token_pos)
            for i in range(len(token_pos) - 1):
                token_i, pos_i = token_pos[i].split(split_token)
                token_j, pos_j = token_pos[i + 1].split(split_token)
                cnt_transition[pos_i][pos_j] += 1
                cnt_emission[pos_i][token_i] += 1
            else:
                token_i, pos_i = token_pos[-1].split(split_token)
                cnt_emission[pos_i][token_i] += 1

        return cnt_transition, cnt_emission

    def _get_prob_dict(
        self, cnt_dict: defaultdict[defaultdict[int]]
    ) -> defaultdict[defaultdict[int]]:
        prob = defaultdict(lambda: defaultdict(int))
        for a, b_cnt in cnt_dict.items():
            num_b = sum([cnt for cnt in b_cnt.values()])
            for b, cnt in b_cnt.items():
                prob[a][b] = cnt / num_b  # MLE
        return prob

    def train(self, dataset: List[str], split_token="_"):
        """
        Compute transition probability and emission probability with MLE
        """
        cnt_transition, cnt_emission = self._get_count_dict(dataset, split_token)
        self.prob_transition = self._get_prob_dict(cnt_transition)
        self.prob_emission = self._get_prob_dict(cnt_emission)

        self.pos = list(cnt_transition.keys())

    def _viterbi(self, tokens: List[str]) -> Tuple[List[List[Optional[str]]], float]:
        prob_path = defaultdict(lambda: defaultdict(int))
        backpointer = defaultdict(lambda: defaultdict(str))
        T = len(tokens)

        # initialization step
        for state in self.pos:
            if (state not in self.prob_transition[self.bos_token]) or (
                tokens[0] not in self.prob_emission[state]
            ):
                continue
            prob_path[1][state] = (
                self.prob_transition[self.bos_token][state]
                * self.prob_emission[state][tokens[0]]
            )
            backpointer[1][state] = self.bos_token

        # recursion step
        for t in range(2, T + 1):
            p_max = 0
            current = None
            pointer = None
            for state in self.pos:
                for state_ in self.pos:
                    if (state not in self.prob_transition[state_]) or (
                        tokens[t - 1] not in self.prob_emission[state]
                    ):
                        continue
                    temp = (
                        prob_path[t - 1][state_]
                        * self.prob_transition[state_][state]
                        * self.prob_emission[state][tokens[t - 1]]
                    )
                    if temp >= p_max:
                        p_max = temp
                        current = state
                        pointer = state_
            else:
                prob_path[t][current] = p_max
                backpointer[t][current] = pointer
        else:
            prob_bestpath = 0
            pointer = None
            for state in self.pos:
                if state not in prob_path[T]:
                    continue
                if prob_path[T][state] > prob_bestpath:
                    prob_bestpath = prob_path[T][state]
                    pointer = state
            bestpath_reversed = [pointer]
            for t in range(T, 1, -1):
                if backpointer[t][pointer] == "":
                    bestpath_reversed.append(None)
                else:
                    bestpath_reversed.append(backpointer[t][pointer])
                pointer = backpointer[t][pointer]

        return list(reversed(bestpath_reversed)), prob_bestpath

    def _get_metrics(self, TP: int, FN: int, FP: int) -> Tuple[float, float, float]:
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * prec * recall / (prec + recall)
        return prec, recall, f1

    def _get_scores(
        self, y_test: List[str], y_pred: List[str]
    ) -> defaultdict[defaultdict[int]]:
        assert len(y_test) == len(y_pred)

        scores = defaultdict(lambda: defaultdict(int))
        for pos_test, pos_pred in zip(y_test, y_pred):
            if pos_test == pos_pred:
                scores[pos_pred]["TP"] += 1
            else:
                scores[pos_test]["FN"] += 1
                scores[pos_pred]["FP"] += 1
        return scores

    def _get_ppl(self, prob: float, N: int) -> float:
        return (1 / prob) ** (1 / N)

    def evaluate(
        self, dataset: List[str], split_token="_", return_each_metric=False
    ) -> Tuple:
        """
        Return macro precision, macro recall, macro f-score and the average of PPL.
        The average of PPL is calculated without the sentences where the path probabilities are assigned as zeros
        """

        macro_scores = defaultdict(lambda: defaultdict(int))
        macro_metrics = defaultdict(lambda: defaultdict(int))
        ppl = []
        cnt_null = 0
        for line in dataset:
            tokens = []
            y_test = []
            for token_pos in line.split():
                token, pos = token_pos.split(split_token)
                tokens.append(token)
                y_test.append(pos)
            y_pred, prob = self._viterbi(tokens)
            if prob > 0:
                ppl.append(self._get_ppl(prob, len(tokens)))
            else:
                cnt_null += 1
            scores = self._get_scores(y_test, y_pred)
            for pos, score_cnt in scores.items():
                for score, cnt in score_cnt.items():
                    macro_scores[pos][score] += cnt
        for pos in macro_scores.keys():
            if "TP" not in macro_scores[pos]:
                continue
            prec, recall, f1 = self._get_metrics(
                macro_scores[pos]["TP"],
                macro_scores[pos]["FN"],
                macro_scores[pos]["FP"],
            )
            macro_metrics[pos]["Precision"] = prec
            macro_metrics[pos]["Recall"] = recall
            macro_metrics[pos]["f-score"] = f1
        ppl_average = sum(ppl) / len(ppl)

        ave_prec = sum(
            [macro_metrics[pos]["Precision"] for pos in macro_scores.keys()]
        ) / len(macro_scores.keys())
        ave_recall = sum(
            [macro_metrics[pos]["Recall"] for pos in macro_scores.keys()]
        ) / len(macro_scores.keys())
        ave_f1 = sum(
            [macro_metrics[pos]["f-score"] for pos in macro_scores.keys()]
        ) / len(macro_scores.keys())

        print(f"The path probabilities of {cnt_null} sentences are assigned as zeros.")
        print(
            f"Precision: {ave_prec}, Recall: {ave_recall}, f-score: {ave_f1}, Average of PPL: {ppl_average}"
        )
        if return_each_metric:
            return ave_prec, ave_recall, ave_f1, ppl_average, macro_metrics
        else:
            return ave_prec, ave_recall, ave_f1, ppl_average

    def tag(self, text: str, split_token="_") -> Tuple[str, float]:
        """
        Return a sequence where POS is assigned to each token, with the specified split token.
        None is assigned when the model can not predict any POS.
        """
        tokens = text.split()
        result, prob = self._viterbi(tokens)
        return (
            " ".join(
                [f"{tokens[i]}{split_token}{pos}" for i, pos in enumerate(result)]
            ),
            prob,
        )


def main(args):
    random.seed(args.seed)
    train_data = load_dataset(args.train_file_path, encoding=args.encoding)
    test_data = load_dataset(args.test_file_path, encoding=args.encoding)
    tagger = HMMTagger()
    tagger.train(train_data, args.split_token)
    if args.mode == "eval":
        _ = tagger.evaluate(test_data, args.split_token)
    else:
        sample = random.choice(test_data)
        text = " ".join(
            [token_pos.split(args.split_token)[0] for token_pos in sample.split()]
        )
        print("Sentence:")
        print(text)
        print("Prediction:")
        print(tagger.tag(text, args.split_token))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="data/wiki-en-train.norm_pos")
    parser.add_argument("--test_file_path", default="data/wiki-en-test.norm_pos")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--split_token", default="_")
    parser.add_argument("--mode", default="eval", choices=["eval", "pred"])
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
