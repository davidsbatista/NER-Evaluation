from copy import deepcopy
from collections import namedtuple, defaultdict

TypelessEntity = namedtuple("Entity", "start_offset end_offset")
Entity = namedtuple("Entity", "e_type start_offset end_offset")


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, tag in enumerate(tokens):

        token_tag = tag

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:]:
            end_offset = offset - 1
            named_entities.append(Entity(ent_type, offset - 1, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, offset))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities):
    """
    Computes the metrics for the full-named entity: correct, incorrect, partial, spurius;

    under four different evaluation schema: strict matching, exact matching, partial matching
    and entity type matching;

    This is defined in SemEval-2013 Task 9.1

    :param true_named_entities: list of predicted named entities represented as Entity named-tuple
    :param pred_named_entities: list of true named entities represented as Entity named-tuple
    :return:
    """
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    evaluation = {'strict': deepcopy(eval_metrics),
                  'exact_matching': deepcopy(eval_metrics),
                  'partial_matching': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics)}

    # if no predictions are made, return precision and recall at 0
    if len(pred_named_entities) == 0:
        eval_metrics.update({
            'precision': 0,
            'recall': 0,
            'possible': 0,
            'actual': 0
        })
        evaluation['strict'] = deepcopy(eval_metrics)
        evaluation['exact_matching'] = deepcopy(eval_metrics)
        evaluation['partial_matching'] = deepcopy(eval_metrics)
        evaluation['ent_type'] = deepcopy(eval_metrics)

        return evaluation

    ############################################################
    # 'partial' and 'exact_matching' don't consider entity type
    ############################################################

    # (re)build typeless Entity named-tuples
    # TODO: this could be improved, someway to re-use the already defined Entity named-tuple
    true_typeless = []
    pred_typeless = []
    for e in true_named_entities:
        true_typeless.append(TypelessEntity(e.start_offset, e.end_offset))
    for e in pred_named_entities:
        pred_typeless.append(TypelessEntity(e.start_offset, e.end_offset))

    # go through each predicted named-entity
    true_which_overlapped_with_pred = []  # keep track of entities that overlapped
    for pred in pred_typeless:
        found_overlap = False

        # check if there's an exact match
        if pred in true_typeless:
            evaluation['exact_matching']['correct'] += 1
            evaluation['partial_matching']['correct'] += 1
            true_which_overlapped_with_pred.append(pred)

        else:
            # check for overlaps with any of the true entities
            for true in true_typeless:
                if pred.start_offset <= true.end_offset and true.start_offset <= pred.end_offset:
                    true_which_overlapped_with_pred.append(true)
                    evaluation['exact_matching']['incorrect'] += 1
                    evaluation['partial_matching']['partial'] += 1
                    found_overlap = True
                    break

            # count over-generated entities
            if not found_overlap:
                evaluation['exact_matching']['spurius'] += 1
                evaluation['partial_matching']['spurius'] += 1

    # count missed entities
    for true in true_typeless:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            evaluation['exact_matching']['missed'] += 1
            evaluation['partial_matching']['missed'] += 1

    ##############################################
    # 'strict' and 'ent_type' consider entity type
    ##############################################
    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # check if there's an exact match, i.e.: boundary and entity type match
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
        else:
            # check for overlaps with any of the true entities
            for true in true_named_entities:
                # check for an exact boundary match but with a different e_type
                if true.start_offset == pred.start_offset and \
                                true.end_offset == pred.end_offset and \
                                true.e_type != pred.e_type:
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap (not exact boundary match) with true entities
                elif pred.start_offset <= true.end_offset \
                        and true.start_offset <= pred.end_offset:
                    true_which_overlapped_with_pred.append(true)
                    if pred.e_type == true.e_type:
                        # overlap and with same e_type:
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        found_overlap = True
                        break
                    else:
                        # overlap with different e_type
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        found_overlap = True
                        break

            # count over-generated entities
            if not found_overlap:
                evaluation['strict']['spurius'] += 1
                evaluation['ent_type']['spurius'] += 1

    # count missed entities
    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1

    # Compute 'possible', 'actual', according to SemEval-2013 Task 9.1
    for eval_type in ['strict', 'exact_matching', 'partial_matching', 'ent_type']:
        correct = evaluation[eval_type]['correct']
        incorrect = evaluation[eval_type]['incorrect']
        partial = evaluation[eval_type]['partial']
        missed = evaluation[eval_type]['missed']
        spurius = evaluation[eval_type]['spurius']

        # possible: nr. annotations in the gold-standard which contribute to the final score
        evaluation[eval_type]['possible'] = correct + incorrect + partial + missed

        # actual: number of annotations produced by the system
        evaluation[eval_type]['actual'] = correct + incorrect + partial + spurius

        actual = evaluation[eval_type]['actual']
        possible = evaluation[eval_type]['possible']

        if eval_type == 'partial_matching':
            precision = (correct + 0.5 * partial) / float(actual)
            if possible == 0:
                recall = 0
            else:
                recall = (correct + 0.5 * partial) / float(possible)

        else:
            precision = correct / float(actual)
            if possible == 0:
                recall = 0
            else:
                recall = correct / float(possible)

        evaluation[eval_type]['precision'] = precision
        evaluation[eval_type]['recall'] = recall

    return evaluation


def compute_metrics_by_type(true_y, y_pred, entity_types):
    """
    computes the metrics aggregated by named-entity type for the full-named entities under
    four different evaluation schema and entity type matching for one given message

    :param true_y: a list of true labels
    :param y_pred: a list of predicted labels
    :param entity_types: the entity types to be considered
    :return:
    """
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    eval_metrics_by_ent_type = {key: deepcopy(eval_metrics) for key in entity_types}
    all_results = {'strict': deepcopy(eval_metrics_by_ent_type),
                   'exact_matching': deepcopy(eval_metrics_by_ent_type),
                   'partial_matching': deepcopy(eval_metrics_by_ent_type),
                   'ent_type': deepcopy(eval_metrics_by_ent_type)}

    for true, pred in zip(true_y, y_pred):
        # collect predicted and true named-entities into lists of Entity named-tuples
        # and aggregated them by entity type
        true_named_entities = collect_named_entities(true)
        pred_named_entities = collect_named_entities(pred)

        true_named_entities_type = defaultdict(list)
        pred_named_entities_type = defaultdict(list)

        for true in true_named_entities:
            true_named_entities_type[true.e_type].append(true)

        for pred in pred_named_entities:
            pred_named_entities_type[pred.e_type].append(pred)

        # compute metrics for each entity type, and for the 4 different evaluation schemas:
        # 'strict', 'exact_matching', 'partial_matching' and 'ent_type'
        for e_type in true_named_entities_type.keys():

            results = compute_metrics(true_named_entities_type[e_type],
                                      pred_named_entities_type[e_type])

            # collect the results for all 4 evaluation types
            for eval_type in ['strict', 'exact_matching', 'partial_matching', 'ent_type']:
                # totals aggregated by entity type
                all_results[eval_type][e_type]['correct'] += results[eval_type]['correct']
                all_results[eval_type][e_type]['incorrect'] += results[eval_type]['incorrect']
                all_results[eval_type][e_type]['partial'] += results[eval_type]['partial']
                all_results[eval_type][e_type]['missed'] += results[eval_type]['missed']
                all_results[eval_type][e_type]['spurius'] += results[eval_type]['spurius']

    return all_results