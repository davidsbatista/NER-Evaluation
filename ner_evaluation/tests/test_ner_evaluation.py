from ner_evaluation.ner_eval import Entity
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import find_overlap


def test_collect_named_entities_same_type_in_sequence():
    tags = ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O']
    result = collect_named_entities(tags)
    expected = [Entity(e_type='LOC', start_offset=1, end_offset=2),
                Entity(e_type='LOC', start_offset=3, end_offset=4)]
    assert result == expected

def test_collect_named_entities_entity_goes_until_last_token():
    tags = ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC']
    result = collect_named_entities(tags)
    expected = [Entity(e_type='LOC', start_offset=1, end_offset=2),
                Entity(e_type='LOC', start_offset=3, end_offset=4)]
    assert result == expected

def test_compute_metrics_case_1():
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
                           'spurious': 1,
                           'possible': 6,
                           'actual': 6,
                           'precision': 0.3333333333333333,
                           'recall': 0.3333333333333333},
                'ent_type': {'correct': 3,
                             'incorrect': 2,
                             'partial': 0,
                             'missed': 1,
                             'spurious': 1,
                             'possible': 6,
                             'actual': 6,
                             'precision': 0.5,
                             'recall': 0.5},
                'partial': {'correct': 3,
                            'incorrect': 0,
                            'partial': 2,
                            'missed': 1,
                            'spurious': 1,
                            'possible': 6,
                            'actual': 6,
                            'precision': 0.6666666666666666,
                            'recall': 0.6666666666666666},
                'exact': {'correct': 3,
                          'incorrect': 2,
                          'partial': 0,
                          'missed': 1,
                          'spurious': 1,
                          'possible': 6,
                          'actual': 6,
                          'precision': 0.5,
                          'recall': 0.5}
                }

    assert results == expected


def test_compute_metrics_agg_scenario_3():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = []

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]


def test_compute_metrics_agg_scenario_2():

    true_named_entities = []

    pred_named_entities = [Entity('PER', 59, 69)]

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]


def test_compute_metrics_agg_scenario_5():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('PER', 57, 69)]

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 1,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]

def test_compute_metrics_agg_scenario_4():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('LOC', 59, 69)]

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        },
        'LOC': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]

def test_compute_metrics_agg_scenario_1():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('PER', 59, 69)]

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]

def test_compute_metrics_agg_scenario_6():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('LOC', 54, 69)]

    results, results_agg = compute_metrics(true_named_entities, pred_named_entities)

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 1,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        },
        'LOC': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0
            }
        }
    }

    assert results_agg["PER"] == expected_agg["PER"]
    assert results_agg["LOC"] == expected_agg["LOC"]

def test_find_overlap_no_overlap():

    pred_entity = Entity('LOC', 1, 10)
    true_entity = Entity('LOC', 11, 20)

    pred_range = range(pred_entity.start_offset, pred_entity.end_offset)
    true_range = range(true_entity.start_offset, true_entity.end_offset)

    pred_set = set(pred_range)
    true_set = set(true_range)

    intersect = find_overlap(pred_set, true_set)

    assert not intersect

def test_find_overlap_total_overlap():

    pred_entity = Entity('LOC', 10, 22)
    true_entity = Entity('LOC', 11, 20)

    pred_range = range(pred_entity.start_offset, pred_entity.end_offset)
    true_range = range(true_entity.start_offset, true_entity.end_offset)

    pred_set = set(pred_range)
    true_set = set(true_range)

    intersect = find_overlap(pred_set, true_set)

    assert intersect

def test_find_overlap_start_overlap():

    pred_entity = Entity('LOC', 5, 12)
    true_entity = Entity('LOC', 11, 20)

    pred_range = range(pred_entity.start_offset, pred_entity.end_offset)
    true_range = range(true_entity.start_offset, true_entity.end_offset)

    pred_set = set(pred_range)
    true_set = set(true_range)

    intersect = find_overlap(pred_set, true_set)

    assert intersect

def test_find_overlap_end_overlap():

    pred_entity = Entity('LOC', 15, 25)
    true_entity = Entity('LOC', 11, 20)

    pred_range = range(pred_entity.start_offset, pred_entity.end_offset)
    true_range = range(true_entity.start_offset, true_entity.end_offset)

    pred_set = set(pred_range)
    true_set = set(true_range)

    intersect = find_overlap(pred_set, true_set)

    assert intersect
