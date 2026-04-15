import unittest

# ייבוא הפונקציות הרלוונטיות מהקובץ הראשי
from algorithms import (
    voting_rule_selector,
    bagging_with_vrs,
    borda_count_rule,
    copeland_rule,
    plurality_rule
)

class MockModel:
    def __init__(self, predictions_dict: dict):
        self.predictions_dict = predictions_dict

    def predict(self, instance: str):
        return self.predictions_dict.get(instance, [])

    def fit(self, dataset):
        pass


class TestLabelRankingEnsemblesUpdated(unittest.TestCase):

    def test_vrs_example_1_plurality_wins(self):
        """
        דוגמא ראשונה: יצרנו מצב שבו Borda ו-Copeland טועים לגמרי בדירוג,
        אבל Plurality מצליח לשמור על המקום הראשון.
        במצב כזה Plurality ינצח ויקבל את מדד הטאו הגבוה ביותר.
        """
        validation_set = [
            ("v1", ["A", "B", "C", "D"]),
            ("v2", ["B", "A", "D", "C"]),
            ("v3", ["C", "D", "B", "A"])
        ]

        # עדכון התחזיות כדי להבטיח ש-Plurality ינצח מתמטית
        m1 = MockModel({"v1": ["A", "C", "D", "B"], "v2": ["B", "D", "C", "A"], "v3": ["C", "B", "A", "D"]})
        m2 = MockModel({"v1": ["A", "D", "C", "B"], "v2": ["B", "C", "D", "A"], "v3": ["C", "A", "B", "D"]})
        m3 = MockModel({"v1": ["B", "C", "D", "A"], "v2": ["A", "D", "C", "B"], "v3": ["D", "B", "A", "C"]})

        models = [m1, m2, m3]
        voting_rules = [borda_count_rule, copeland_rule, plurality_rule]

        best_rule = voting_rule_selector(models, validation_set, voting_rules)
        self.assertEqual(best_rule.__name__, plurality_rule.__name__)


    def test_vrs_example_2_copeland_wins(self):
        """
        דוגמא שנייה: יצרנו מצב של 'פרדוקס קונדורסה' קלאסי.
        במצב זה Borda נותן תיקו בין המקומות הראשונים, אך Copeland מצליח
        לשבור את התיקו בעזרת ניצחונות ראש-בראש ולהגיע לדיוק מושלם (1.0).
        """
        validation_set = [
            ("v1", ["X", "Y", "Z"]),
            ("v2", ["Y", "X", "Z"])
        ]

        # מודלים שמותאמים להכשיל את Borda ולהעצים את Copeland
        m1 = MockModel({"v1": ["X", "Y", "Z"], "v2": ["Y", "X", "Z"]})
        m2 = MockModel({"v1": ["X", "Y", "Z"], "v2": ["Y", "X", "Z"]})
        m3 = MockModel({"v1": ["Y", "Z", "X"], "v2": ["X", "Z", "Y"]})

        models = [m1, m2, m3]
        voting_rules = [borda_count_rule, copeland_rule]

        best_rule = voting_rule_selector(models, validation_set, voting_rules)
        self.assertEqual(best_rule.__name__, copeland_rule.__name__)


    def test_bvrs_example_1_food_preferences(self):
        """
        דוגמא ראשונה (BVRS): העדפות מזון.
        הבדיקה עברה אצלך בהצלחה ומשאירה את Borda כמנצח עם דיוק 1.0.
        """
        m1 = MockModel({
            "v1": ["Pizza", "Pasta", "Salad", "Soup"],
            "v2": ["Pasta", "Pizza", "Salad", "Soup"],
            "T1": ["Pizza", "Pasta", "Soup", "Salad"],
            "T2": ["Pasta", "Salad", "Pizza", "Soup"]
        })
        m2 = MockModel({
            "v1": ["Pasta", "Pizza", "Salad", "Soup"],
            "v2": ["Salad", "Pasta", "Pizza", "Soup"],
            "T1": ["Pizza", "Pasta", "Soup", "Salad"],
            "T2": ["Pasta", "Salad", "Pizza", "Soup"]
        })
        m3 = MockModel({
            "v1": ["Pizza", "Salad", "Pasta", "Soup"],
            "v2": ["Pizza", "Salad", "Pasta", "Soup"],
            "T1": ["Pizza", "Pasta", "Soup", "Salad"],
            "T2": ["Pasta", "Salad", "Pizza", "Soup"]
        })

        models_queue = [m1, m2, m3]
        def mock_factory():
            return models_queue.pop(0)

        train_set = [
            ("dummy", []), ("dummy", []), ("dummy", []), ("dummy", []), ("dummy", []),
            ("v1", ["Pizza", "Pasta", "Salad", "Soup"]),
            ("v2", ["Pasta", "Pizza", "Salad", "Soup"])
        ]
        train_val_ratio = 5 / 7.0

        test_set = [
            ("T1", ["Pizza", "Pasta", "Soup", "Salad"]),
            ("T2", ["Pasta", "Salad", "Pizza", "Soup"])
        ]
        voting_rules = [borda_count_rule, copeland_rule, plurality_rule]

        test_tau = bagging_with_vrs(train_set, test_set, mock_factory, 3, train_val_ratio, voting_rules)
        self.assertAlmostEqual(test_tau, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()