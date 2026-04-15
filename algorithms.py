"""
Paper: Beyond majority: Label ranking ensembles based on voting rules
Authors: Havi Werbin-Ofir, Lihi Dery, Erez Shmueli
Link: https://doi.org/10.1016/j.eswa.2019.06.022
Students: [שחר אחדות]
"""

import logging
import random
import math
from typing import List, Tuple, Callable, Any, Dict, Union
from collections import Counter
import scipy.stats as stats

# הגדרת הגדרות הלוג
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Type Aliases ---
# דירוג יכול כעת להכיל מחרוזת (תווית רגילה) או קבוצה (set) המייצגת תיקו בין כמה תוויות
RankingElement = Union[str, set]
Ranking = List[RankingElement]
Instance = Any  # Can be a feature vector or any object representing the features
Dataset = List[Tuple[Instance, Ranking]]
VotingRule = Callable[[List[Ranking]], Ranking]

class Model:
    """
    A dummy interface for a base learner model.
    """
    def predict(self, instance: Instance) -> Ranking:
        pass

    def fit(self, dataset: Dataset):
        pass


# --- Helper Functions for Kendall Tau and Ties ---

def kendall_tau_distance(ranking_a: Ranking, ranking_b: Ranking) -> float:
    """
    Calculates the Kendall-tau distance between two strict rankings (no ties).
    Returns a float between -1.0 and 1.0.

    >>> kendall_tau_distance(['A', 'B', 'C'], ['A', 'B', 'C'])
    1.0
    """
    tau, p_value = stats.kendalltau(ranking_a, ranking_b)
    if math.isnan(tau):
        return 0.0
    return float(tau)

def get_ranks(ranking: Ranking, elements: List[str]) -> Dict[str, float]:
    """
    ממירה דירוג (שיכול להכיל קבוצות של תיקו) למילון של ציוני מיקום עבור kendall_tau_b.
    """
    ranks = {}
    current_rank = 1
    for item in ranking:
        if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
            for sub_item in item:
                ranks[sub_item] = current_rank
            current_rank += len(item)
        else:
            ranks[item] = current_rank
            current_rank += 1

    for e in elements:
        if e not in ranks:
            ranks[e] = current_rank

    return ranks

def kendall_tau_b(ranking_a: Ranking, ranking_b: Ranking) -> float:
    """
    Calculates the Kendall-tau-b distance handling ties.
    Returns a float between -1.0 and 1.0.
    """
    elements = set()
    for r in [ranking_a, ranking_b]:
        for item in r:
            if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
                elements.update(item)
            else:
                elements.add(item)

    elements_list = list(elements)
    if not elements_list:
        return 0.0

    ranks_a = get_ranks(ranking_a, elements_list)
    ranks_b = get_ranks(ranking_b, elements_list)

    n = len(elements_list)
    n0 = n * (n - 1) / 2
    nc = nd = n1 = n2 = 0

    for i in range(n):
        for j in range(i + 1, n):
            e_i, e_j = elements_list[i], elements_list[j]

            sign_a = (ranks_a[e_j] > ranks_a[e_i]) - (ranks_a[e_j] < ranks_a[e_i])
            sign_b = (ranks_b[e_j] > ranks_b[e_i]) - (ranks_b[e_j] < ranks_b[e_i])

            if sign_a == 0: n1 += 1
            if sign_b == 0: n2 += 1

            if sign_a * sign_b > 0:
                nc += 1
            elif sign_a * sign_b < 0:
                nd += 1

    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return 0.0
    return (nc - nd) / denom

def calculate_tau(ranking_a: Ranking, ranking_b: Ranking) -> float:
    """
    פונקציית הנתב: בודקת האם יש שוויון (Tie) באחד הדירוגים.
    אם יש שוויון -> מפעילה kendall_tau_b.
    אם אין שוויון -> מפעילה את kendall_tau הרגיל.
    """
    has_tie = False
    for r in [ranking_a, ranking_b]:
        for item in r:
            if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
                has_tie = True
                break
        if has_tie:
            break

    if has_tie:
        logging.debug("Tie detected in rankings. Using kendall_tau_b.")
        return kendall_tau_b(ranking_a, ranking_b)
    else:
        logging.debug("Strict rankings detected. Using regular kendall_tau_distance.")
        return kendall_tau_distance(ranking_a, ranking_b)


# --- Data and Rule Helpers ---

def scores_to_ranking(scores: Dict[str, float]) -> Ranking:
    """
    הופכת מילון ציונים לדירוג ומייצרת 'תיקו' (set) כשיש שוויון.
    """
    score_to_cands = {}
    for cand, score in scores.items():
        s = round(score, 5)
        score_to_cands.setdefault(s, set()).add(cand)

    sorted_scores = sorted(score_to_cands.keys(), reverse=True)
    ranking = []
    for score in sorted_scores:
        cands = score_to_cands[score]
        if len(cands) == 1:
            ranking.append(list(cands)[0])
        else:
            ranking.append(cands)
    return ranking

def split_data(dataset: Dataset, ratio: float) -> Tuple[Dataset, Dataset]:
    logging.debug(f"Splitting dataset of size {len(dataset)} with train ratio {ratio}")
    split_index = int(len(dataset) * ratio)
    return dataset[:split_index], dataset[split_index:]

def bootstrap_sample(dataset: Dataset) -> Dataset:
    logging.debug("Creating a bootstrap sample with replacement")
    return random.choices(dataset, k=len(dataset))

def get_candidates(rankings: List[Ranking]) -> List[str]:
    candidates = set()
    for ranking in rankings:
        for item in ranking:
            if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
                candidates.update(item)
            else:
                candidates.add(item)
    return list(candidates)


# --- Voting Rules ---

def borda_count_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Borda Count rule")
    candidates = get_candidates(rankings)
    scores = {c: 0.0 for c in candidates}
    n = len(candidates)

    for ranking in rankings:
        current_pos = 0
        for item in ranking:
            if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
                k = len(item)
                total_points = sum(n - 1 - (current_pos + i) for i in range(k))
                avg_points = total_points / k
                for cand in item:
                    scores[cand] += avg_points
                current_pos += k
            else:
                scores[item] += (n - 1 - current_pos)
                current_pos += 1

    return scores_to_ranking(scores)

def copeland_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Copeland rule")
    candidates = get_candidates(rankings)
    scores = {c: 0.0 for c in candidates}

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            wins_c1 = wins_c2 = 0

            for ranking in rankings:
                ranks = get_ranks(ranking, [c1, c2])
                if ranks[c1] < ranks[c2]:
                    wins_c1 += 1
                elif ranks[c2] < ranks[c1]:
                    wins_c2 += 1

            if wins_c1 > wins_c2:
                scores[c1] += 1
                scores[c2] -= 1
            elif wins_c2 > wins_c1:
                scores[c2] += 1
                scores[c1] -= 1

    return scores_to_ranking(scores)

def plurality_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Plurality rule")
    candidates = get_candidates(rankings)
    scores = {c: 0.0 for c in candidates}

    for ranking in rankings:
        if ranking:
            first_place = ranking[0]
            if isinstance(first_place, (set, list, tuple)) and not isinstance(first_place, str):
                point = 1.0 / len(first_place)
                for c in first_place:
                    scores[c] += point
            else:
                scores[first_place] += 1.0

    return scores_to_ranking(scores)

def veto_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Veto rule")
    candidates = get_candidates(rankings)
    scores = {c: 0.0 for c in candidates}

    for ranking in rankings:
        if ranking:
            last_place = ranking[-1]
            last_place_set = last_place if isinstance(last_place, (set, list, tuple)) and not isinstance(last_place, str) else {last_place}

            for candidate in candidates:
                if candidate not in last_place_set:
                    scores[candidate] += 1.0

    return scores_to_ranking(scores)

def stv_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying STV rule")
    candidates = get_candidates(rankings)
    active_candidates = set(candidates)
    result_ranking = []

    working_rankings = []
    for r in rankings:
        new_r = []
        for item in r:
            if isinstance(item, (set, list, tuple)) and not isinstance(item, str):
                new_r.extend(list(item))
            else:
                new_r.append(item)
        working_rankings.append(new_r)

    while active_candidates:
        first_place_counts = {c: 0 for c in active_candidates}
        for r in working_rankings:
            if r:
                first_place_counts[r[0]] += 1

        winner = max(active_candidates, key=lambda c: first_place_counts[c])
        result_ranking.append(winner)
        active_candidates.remove(winner)

        for r in working_rankings:
            if winner in r:
                r.remove(winner)

    return result_ranking

def maximin_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Maximin rule")
    candidates = get_candidates(rankings)
    scores = {c: float('inf') for c in candidates}

    for c1 in candidates:
        for c2 in candidates:
            if c1 == c2:
                continue

            wins_c1 = 0
            for ranking in rankings:
                ranks = get_ranks(ranking, [c1, c2])
                if ranks[c1] < ranks[c2]:
                    wins_c1 += 1

            if wins_c1 < scores[c1]:
                scores[c1] = wins_c1

    return scores_to_ranking(scores)

def modal_ranking_rule(rankings: List[Ranking]) -> Ranking:
    logging.debug("Applying Modal Ranking rule")
    safe_rankings = []
    for r in rankings:
        safe_r = tuple(frozenset(item) if isinstance(item, set) else item for item in r)
        safe_rankings.append(safe_r)

    counts = Counter(safe_rankings)
    most_common = counts.most_common(1)[0][0]

    return [set(item) if isinstance(item, frozenset) else item for item in most_common]


# --- Algorithm 1: Voting Rule Selector (VRS) ---

def voting_rule_selector(models: List[Model], validation_set: Dataset, voting_rules: List[VotingRule]) -> VotingRule:
    """
    Algorithm 1 (VRS): Iterates over all voting rules and selects the one
    that achieves the highest Kendall-tau accuracy on the validation set.
    """
    logging.info(f"Starting VRS with {len(voting_rules)} rules and {len(validation_set)} validation instances.")

    best_rule = None
    best_average_tau = -float('inf')

    for rule in voting_rules:
        logging.debug(f"Evaluating rule: {rule.__name__}")
        total_tau = 0.0

        for instance, true_ranking in validation_set:
            predictions = [model.predict(instance) for model in models]
            aggregated_ranking = rule(predictions)

            # שימוש בפונקציית הנתב שמחליטה באיזה קנדל-טאו להשתמש
            tau = calculate_tau(aggregated_ranking, true_ranking)
            total_tau += tau

        average_tau = total_tau / len(validation_set)
        logging.info(f"Rule {rule.__name__} achieved average tau: {average_tau:.4f}")

        if average_tau > best_average_tau:
            best_average_tau = average_tau
            best_rule = rule

    logging.info(f"VRS selected best rule: {best_rule.__name__}")
    return best_rule


# --- Algorithm 2: Bagging with VRS (BVRS) ---

def bagging_with_vrs(train_set: Dataset, test_set: Dataset,
                     base_learner_factory: Callable[[], Model],
                     num_bags: int, train_val_ratio: float,
                     voting_rules: List[VotingRule]) -> float:
    """
    Algorithm 2 (BVRS): Manages the Bagging process, invokes VRS to choose
    the best voting rule, and returns the average accuracy on the test set.
    """
    logging.info(f"Starting BVRS with {num_bags} bags.")

    t_prime, validation_set = split_data(train_set, train_val_ratio)
    models = []

    for i in range(num_bags):
        logging.info(f"Training bag {i+1}/{num_bags}")
        bag_sample = bootstrap_sample(t_prime)
        model = base_learner_factory()
        model.fit(bag_sample)
        models.append(model)

    best_voting_rule = voting_rule_selector(models, validation_set, voting_rules)

    logging.info("Evaluating on the test set using the selected voting rule.")
    total_test_tau = 0.0

    for instance, true_ranking in test_set:
        predictions = [model.predict(instance) for model in models]
        final_ranking = best_voting_rule(predictions)

        # שימוש בפונקציית הנתב שמחליטה באיזה קנדל-טאו להשתמש
        tau = calculate_tau(final_ranking, true_ranking)
        total_test_tau += tau

    average_test_tau = total_test_tau / len(test_set)
    logging.info(f"Final Test Average Kendall-Tau: {average_test_tau:.4f}")

    return average_test_tau