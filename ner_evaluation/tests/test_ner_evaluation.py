from ner_evaluation.ner_eval import Entity
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import find_overlap
from ner_evaluation.ner_eval import compute_actual_possible
from ner_evaluation.ner_eval import compute_precision_recall
from ner_evaluation.ner_eval import compute_precision_recall_wrapper
from ner_evaluation.ner_eval import get_tags


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


def test_collect_named_entities_no_entity():
    tags = ['O', 'O', 'O', 'O', 'O']
    result = collect_named_entities(tags)
    expected = []
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

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'LOC', 'MISC']
    )

    results = compute_precision_recall_wrapper(results)

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

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0,
                'actual': 0,
                'possible': 1,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0,
                'actual': 0,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0,
                'actual': 0,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 1,
                'spurious': 0,
                'actual': 0,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']


def test_compute_metrics_agg_scenario_2():

    true_named_entities = []

    pred_named_entities = [Entity('PER', 59, 69)]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1,
                'actual': 1,
                'possible': 0,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1,
                'actual': 1,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1,
                'actual': 1,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 1,
                'actual': 1,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']


def test_compute_metrics_agg_scenario_5():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('PER', 57, 69)]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 1,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']


def test_compute_metrics_agg_scenario_4():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('LOC', 59, 69)]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'LOC']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            }
        },
        'LOC': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']

    assert results_agg['LOC'] == expected_agg['LOC']


def test_compute_metrics_agg_scenario_1():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('PER', 59, 69)]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 1,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']


def test_compute_metrics_agg_scenario_6():

    true_named_entities = [Entity('PER', 59, 69)]

    pred_named_entities = [Entity('LOC', 54, 69)]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'LOC']
    )

    expected_agg = {
        'PER': {
            'strict': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 1,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 1,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 1,
                'possible': 1,
                'precision': 0,
                'recall': 0,
            }
        },
        'LOC': {
            'strict': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
                },
            'ent_type': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'partial': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            },
            'exact': {
                'correct': 0,
                'incorrect': 0,
                'partial': 0,
                'missed': 0,
                'spurious': 0,
                'actual': 0,
                'possible': 0,
                'precision': 0,
                'recall': 0,
            }
        }
    }

    assert results_agg['PER']['strict'] == expected_agg['PER']['strict']
    assert results_agg['PER']['ent_type'] == expected_agg['PER']['ent_type']
    assert results_agg['PER']['partial'] == expected_agg['PER']['partial']
    assert results_agg['PER']['exact'] == expected_agg['PER']['exact']

    assert results_agg["LOC"] == expected_agg["LOC"]


def test_compute_metrics_extra_tags_in_prediction():

    true_named_entities = [
        Entity('PER', 50, 52),
        Entity('ORG', 59, 69),
        Entity('ORG', 71, 72),
    ]

    pred_named_entities = [
        Entity('LOC', 50, 52),  # Wrong type
        Entity('ORG', 59, 69),  # Correct
        Entity('MISC', 71, 72), # Wrong type
    ]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'LOC', 'ORG']
    )

    expected = {
        'strict': {
            'correct': 1,
            'incorrect': 2,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'actual': 3,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        },
        'ent_type': {
            'correct': 1,
            'incorrect': 2,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'actual': 3,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        },
        'partial': {
            'correct': 3,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'actual': 3,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        },
        'exact': {
            'correct': 3,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'actual': 3,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        }
    }

    assert results['strict'] == expected['strict']
    assert results['ent_type'] == expected['ent_type']
    assert results['partial'] == expected['partial']
    assert results['exact'] == expected['exact']


def test_compute_metrics_extra_tags_in_true():

    true_named_entities = [
        Entity('PER', 50, 52),
        Entity('ORG', 59, 69),
        Entity('MISC', 71, 72),
    ]

    pred_named_entities = [
        Entity('LOC', 50, 52),  # Wrong type
        Entity('ORG', 59, 69),  # Correct
        Entity('ORG', 71, 72),  # Spurious
    ]

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'LOC', 'ORG']
    )

    expected = {
        'strict': {
            'correct': 1,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 1,
            'actual': 3,
            'possible': 2,
            'precision': 0,
            'recall': 0,
            },
        'ent_type': {
            'correct': 1,
            'incorrect': 1,
            'partial': 0,
            'missed': 0,
            'spurious': 1,
            'actual': 3,
            'possible': 2,
            'precision': 0,
            'recall': 0,
        },
        'partial': {
            'correct': 2,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 1,
            'actual': 3,
            'possible': 2,
            'precision': 0,
            'recall': 0,
        },
        'exact': {
            'correct': 2,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 1,
            'actual': 3,
            'possible': 2,
            'precision': 0,
            'recall': 0,
        }
    }

    assert results['strict'] == expected['strict']
    assert results['ent_type'] == expected['ent_type']
    assert results['partial'] == expected['partial']
    assert results['exact'] == expected['exact']


def test_compute_metrics_no_predictions():

    true_named_entities = [
        Entity('PER', 50, 52),
        Entity('ORG', 59, 69),
        Entity('MISC', 71, 72),
    ]

    pred_named_entities = []

    results, results_agg = compute_metrics(
        true_named_entities, pred_named_entities, ['PER', 'ORG', 'MISC']
    )

    expected = {
        'strict': {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 3,
            'spurious': 0,
            'actual': 0,
            'possible': 3,
            'precision': 0,
            'recall': 0,
            },
        'ent_type': {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 3,
            'spurious': 0,
            'actual': 0,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        },
        'partial': {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 3,
            'spurious': 0,
            'actual': 0,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        },
        'exact': {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 3,
            'spurious': 0,
            'actual': 0,
            'possible': 3,
            'precision': 0,
            'recall': 0,
        }
    }

    assert results['strict'] == expected['strict']
    assert results['ent_type'] == expected['ent_type']
    assert results['partial'] == expected['partial']
    assert results['exact'] == expected['exact']

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


def test_compute_actual_possible():

    results = {
        'correct': 6,
        'incorrect': 3,
        'partial': 2,
        'missed': 4,
        'spurious': 2,
        }

    expected = {
        'correct': 6,
        'incorrect': 3,
        'partial': 2,
        'missed': 4,
        'spurious': 2,
        'possible': 15,
        'actual': 13,
    }

    out = compute_actual_possible(results)

    assert out == expected


def test_compute_precision_recall():

    results = {
        'correct': 6,
        'incorrect': 3,
        'partial': 2,
        'missed': 4,
        'spurious': 2,
        'possible': 15,
        'actual': 13,
        }

    expected = {
        'correct': 6,
        'incorrect': 3,
        'partial': 2,
        'missed': 4,
        'spurious': 2,
        'possible': 15,
        'actual': 13,
        'precision': 0.46153846153846156, 
        'recall': 0.4
    }

    out = compute_precision_recall(results)

    assert out == expected


def test_get_tags():

    sents = [[
        'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 
        'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O',
    ]]

    expected = set(['PER', 'ORG'])

    out = get_tags(sents)

    assert out == expected
