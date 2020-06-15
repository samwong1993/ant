# coding: utf-8
# pylint: disable = invalid-name, W0105
"""Library with tree-based rule training routines of LightGBM."""
from __future__ import absolute_import

import collections
import copy
import warnings
import logging
import time
import random
from operator import attrgetter
from collections import OrderedDict

import numpy as np

from . import callback
from .basic import Booster, Dataset, LightGBMError, _InnerPredictor, global_sync_up_by_sum, global_sync_up_by_min
from .compat import (SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold,
                     string_type, integer_types, range_, zip_)


logger = logging.getLogger("Rule Engine")


def params_check(params):
    objective = params.get('objective')
    if objective != 'binary':
        raise LightGBMError('Only support `binary` objective in sequential_covering mode, but got {} !'.format(objective))


def rule_train(params, train_set, num_boost_round=100,
               valid_sets=None, valid_names=None,
               fobj=None, feval=None, init_model=None,
               feature_name='auto', categorical_feature='auto',
               early_stopping_rounds=None, evals_result=None,
               verbose_eval=True, learning_rates=None,
               keep_training_booster=False, callbacks=None,
               rule_top_n_each_tree=10, positive_shreshold=0.5,
               num_stop_threshold=20, weight_decay=0.8):
    # create predictor first
    params = copy.deepcopy(params)
    params_check(params)
    if fobj is not None:
        params['objective'] = 'none'
    for alias in ["num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees",
                  "num_round", "num_rounds", "num_boost_round", "n_estimators"]:
        if alias in params:
            num_boost_round = int(params.pop(alias))
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            break
    for alias in ["early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change"]:
        if alias in params:
            early_stopping_rounds = int(params.pop(alias))
            warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
            break
    for alias in ["rule_top_n_each_tree"]:
        if alias in params:
            rule_top_n_each_tree = int(params.pop(alias))
            break
    for alias in ["positive_shreshold"]:
        if alias in params:
            positive_shreshold = float(params.pop(alias))
            break
    for alias in ["num_stop_threshold"]:
        if alias in params:
            num_stop_threshold = int(params.get(alias))
            break
    first_metric_only = params.pop('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, string_type):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    # check dataset
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    train_set._update_params(params) \
        ._set_predictor(predictor) \
        .set_feature_name(feature_name) \
        .set_categorical_feature(categorical_feature)

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, string_type):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            # reduce cost for prediction training data
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            if not isinstance(valid_data, Dataset):
                raise TypeError("Training only accepts Dataset object")
            reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append('valid_' + str(i))
    # process callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)

        # Most of legacy advanced options becomes callbacks
    if verbose_eval is True:
        callbacks.add(callback.print_evaluation())
    elif isinstance(verbose_eval, integer_types):
        callbacks.add(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=bool(verbose_eval)))

        if learning_rates is not None:
            callbacks.add(callback.reset_parameter(learning_rate=learning_rates))

    if evals_result is not None:
        callbacks.add(callback.record_evaluation(evals_result))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    # construct booster
    try:
        booster = Booster(params=params, train_set=train_set)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
        for valid_set, name_valid_set in zip_(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    remain_train_data = train_set.num_data()
    train_set.set_weight([1.0] * remain_train_data)
    rule_list = []
    # start training
    for i in range_(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(callback.CallbackEnv(model=booster,
                                    params=params,
                                    iteration=i,
                                    begin_iteration=init_iteration,
                                    end_iteration=init_iteration + num_boost_round,
                                    evaluation_result_list=None))

        booster.update(train_set, fobj=fobj)
        pred_leaf = booster.predict(train_set.get_data(), pred_leaf=True)
        if len(pred_leaf.shape) > 1:
            pred_leaf = pred_leaf[:, -1]
        top_rules_info = compute_top_n_leaf_info(pred_leaf, train_set.get_label(), train_set.get_weight(), rule_top_n_each_tree, positive_shreshold)
        top_leaf_ids = top_rules_info.keys()

        paths, rule_links = booster.get_leaf_path(i, gen_rule_link=True, with_score=False, missing_value=True)
        paths = paths[0]
        rule_links = rule_links[0]
        for leaf_id in top_leaf_ids:
            rule = paths[leaf_id] + ' then {}'.format(round(top_rules_info[leaf_id]['score'], 6))
            rule_list.append({
                'rule_info': top_rules_info[leaf_id],
                'rule_link': rule_links[leaf_id],
                'rule': rule,
                'tree_id': i,
                'rule_id': '{}_{}'.format(i, leaf_id)
            })

        evaluation_result_list = []
        train_set, drop_data_count = compute_weight(train_set, pred_leaf, top_leaf_ids)
        remain_train_data = remain_train_data - drop_data_count
        global_train_set_size = global_sync_up_by_sum(remain_train_data)
        logger.warning('iter {}, remaining {} samples to train in local worker, '
                       '{} global.'.format(i, remain_train_data, int(global_train_set_size)))
        if global_train_set_size <= num_stop_threshold:
            logger.warning('remaining samples is less(equal) than num_stop_threshold({}), stop training'.format(num_stop_threshold))
            break

        # check evaluation result.
        if valid_sets is not None:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=params,
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=evaluation_result_list))
        except callback.EarlyStopException as earlyStopException:
            booster.best_iteration = earlyStopException.best_iteration + 1
            evaluation_result_list = earlyStopException.best_score
            break

    logger.warning('worker is ready to exit.')
    booster.best_score = collections.defaultdict(dict)
    for dataset_name, eval_name, score, _ in evaluation_result_list:
        booster.best_score[dataset_name][eval_name] = score
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string(), False).free_dataset()
    return booster, rule_list


def compute_top_n_leaf_info(preds_leaf, labels, weights, top_n, positive_shreshold=0.5):
    leaf_info = compute_leaf_info(preds_leaf, labels, weights, positive_shreshold)
    top_n = min(top_n, len(leaf_info))
    leaf_info_list = sorted(leaf_info.items(), key=lambda item: item[1]['accuracy'], reverse=True)
    top_rules = OrderedDict()
    for i in range(top_n):
        top_rules[leaf_info_list[i][0]] = leaf_info_list[i][1]
    return top_rules


def compute_leaf_info(preds_leaf, labels, positive_shreshold=0.5):
    leaf_info = {}
    for i in range(len(preds_leaf)):
        leaf_id = preds_leaf[i]
        if leaf_id not in leaf_info:
            leaf_info[leaf_id] = {0: 0., 1: 0.}
        leaf_info[leaf_id][labels[i]] += 1

    for leaf_id in leaf_info:
        global_count_0 = global_sync_up_by_sum(leaf_info[leaf_id][0])
        global_count_1 = global_sync_up_by_sum(leaf_info[leaf_id][1])
        global_count_sum = global_count_1 + global_count_0 + 1e-10
        global_pos_rate = global_count_1 / global_count_sum
        global_leaf_pred_class = 1 if global_pos_rate >= positive_shreshold else 0
        global_accuracy = float(global_count_0) / global_count_sum if global_leaf_pred_class == 0 \
            else float(global_count_1) / global_count_sum
        positive_rate = global_count_1 / global_count_sum
        leaf_info[leaf_id][0] = global_count_0
        leaf_info[leaf_id][1] = global_count_1
        # leaf_info[leaf_id]['accuracy'] = global_accuracy
        leaf_info[leaf_id]['accuracy'] = positive_rate
        leaf_info[leaf_id]['pred'] = global_leaf_pred_class
        leaf_info[leaf_id]['score'] = float(global_count_1) / global_count_sum

    return leaf_info


def compute_weight(origin_dataset, pred_leaf, drop_leaf_id):
    drop_indices = []
    weight = origin_dataset.get_weight()
    for i in range(len(pred_leaf)):
        if pred_leaf[i] in drop_leaf_id and weight[i] > 10e-10:
            drop_indices.append(i)
    drop_data_count = len(drop_indices)
    for idx in drop_indices:
        weight[idx] = 0
    origin_dataset.set_weight(weight)
    return origin_dataset, drop_data_count
