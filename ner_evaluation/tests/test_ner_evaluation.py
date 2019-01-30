import unittest

from ner_evaluation.ner_eval import Entity
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import collect_named_entities


class ner_evaluation(unittest.TestCase):

    def test_collect_named_entities_same_type_in_sequence(self):
        tags = ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O']
        result = collect_named_entities(tags)
        expected = [Entity(e_type='LOC', start_offset=1, end_offset=2),
                    Entity(e_type='LOC', start_offset=3, end_offset=4)]
        self.assertEqual(result, expected)

    def test_collect_named_entities_entity_goes_until_last_token(self):
        tags = ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
        result = collect_named_entities(tags)
        expected = [Entity(e_type='LOC', start_offset=1, end_offset=2),
                    Entity(e_type='LOC', start_offset=3, end_offset=4)]
        self.assertEqual(result, expected)

    def test_compute_metrics_case_1(self):
        true_named_entities = [
            Entity('PER', 59, 69),
            Entity('LOC', 127, 134),
            Entity('LOC', 164, 174),
            Entity('LOC', 197, 205),
            Entity('LOC', 208, 219),
            Entity('MISC', 230, 240)
        ]

        pred_named_entities = [
            Entity('PER', 24, 30),
            Entity('LOC', 124, 134),
            Entity('PER', 164, 174),
            Entity('LOC', 197, 205),
            Entity('LOC', 208, 219),
            Entity('LOC', 225, 243)
        ]

        results, results_agg = compute_metrics(true_named_entities, pred_named_entities)
        expected = {'strict': {'correct': 2,
                               'incorrect': 3,
                               'partial': 0,
                               'missed': 1,
                               'spurius': 1,
                               'possible': 6,
                               'actual': 6,
                               'precision': 0.3333333333333333,
                               'recall': 0.3333333333333333},
                    'ent_type': {'correct': 3,
                                 'incorrect': 2,
                                 'partial': 0,
                                 'missed': 1,
                                 'spurius': 1,
                                 'possible': 6,
                                 'actual': 6,
                                 'precision': 0.5,
                                 'recall': 0.5},
                    'partial': {'correct': 3,
                                'incorrect': 0,
                                'partial': 2,
                                'missed': 1,
                                'spurius': 1,
                                'possible': 6,
                                'actual': 6,
                                'precision': 0.6666666666666666,
                                'recall': 0.6666666666666666},
                    'exact': {'correct': 3,
                              'incorrect': 2,
                              'partial': 0,
                              'missed': 1,
                              'spurius': 1,
                              'possible': 6,
                              'actual': 6,
                              'precision': 0.5,
                              'recall': 0.5}
                    }

        from pprint import pprint

        for k in results:
            print(k)
            pprint(results[k])
            print()
            pprint(expected[k])
            print()
            print("========")

        self.assertDictEqual(results, expected)
