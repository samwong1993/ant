# coding: utf-8
# pylint: disable = invalid-name, W0105
"""Library with tree-based rule training routines of LightGBM."""
from __future__ import absolute_import

import collections
import copy
import warnings
import logging
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
               rule_top_n_each_tree=10, rule_pos_rate_threshold=None,
               num_stop_threshold=20):
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
    for alias in ["rule_pos_rate_threshold"]:
        if alias in params:
            rule_pos_rate_threshold = float(params.pop(alias))
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

    # original_train_set = train_set
    # num_data = len(original_train_set.data)
    # used_indices = [i for i in range(num_data)]
    # train_set = original_train_set.subset(used_indices)

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

    labels = []
    labels.append(train_set.label)
    #datasets_stats 统计数据集中label的个数
    datasets_stats = []
    train_stats = {0: 0, 1: 0}
    for y in train_set.label:
        train_stats[y] += 1
    datasets_stats.append(train_stats)
    if valid_sets:
        for valid_set in valid_sets:
            valid_stats = {0: 0, 1: 0}
            for y in valid_set.label:
                valid_stats[y] += 1
            datasets_stats.append(valid_stats)
        for valid_set in valid_sets:
            labels.append(valid_set.label)

    for i in range(len(datasets_stats)):
        datasets_stats[i][0] = global_sync_up_by_sum(datasets_stats[i][0])
        datasets_stats[i][1] = global_sync_up_by_sum(datasets_stats[i][1])
    train_index = []
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
        #booster.update(valid_sets[0], fobj=fobj)
        booster.update(train_set, fobj=fobj)
        pred_leaf_list = []
        #pred_leaf训练集
        #print(train_set.get_data()) 带有参数的类似csv
        pred_leaf = booster.predict(train_set.get_data(), pred_leaf=True)
        pred_leaf_list.append(pred_leaf)
        if valid_sets:
            for valid_set in valid_sets:
                #print(len(valid_set.get_data()))
                #测试集
                valid_pred_leaf = booster.predict(valid_set.get_data(), pred_leaf=True)
                pred_leaf_list.append(valid_pred_leaf)
        #pred_leaf_list [array([[11, 10, 10, ..., 10, 13, 8]]), array([[11, 5, 13, ..., 5, 5, 6]])]
        #print(len(pred_leaf_list[0]))
        #pred_leaf_list存储训练集测试集结果
        if len(pred_leaf_list[0].shape) > 1:
            for j in range(len(pred_leaf_list)):
                pred_leaf_list[j] = pred_leaf_list[j][:, -1]
            # pred_leaf = pred_leaf[:, -1]
        #pred_leaf_list：array([6, 4, 3, 1, 3, 6, 1, 6, 1, 1, 6, 7, 1, 2, 0, 1, 6, 1, 5, 3, 5, 4, 6]), array([5, 5, 1, ..., 5, 0, 2])]
        select_rule_num = rule_top_n_each_tree
        if i == init_iteration + num_boost_round - 1:
            select_rule_num = -1
        #datasets_stats[{0: 15442.0, 1: 4558.0}, {0: 7922.0, 1: 2078.0}]
        top_rules_info = compute_top_n_leaf_info(pred_leaf_list, datasets_stats, labels, select_rule_num, rule_pos_rate_threshold)
        top_leaf_ids = top_rules_info.keys()
        #print(top_leaf_ids)
        #odict_keys([1, 4, 0, 2, 6, 5, 3, 7])
        #top_rules_info
        #trainset:OrderedDict([(1, [{0: 130, 1: 32, 'recall': 0.007020623080298377, 'precision': 0.19753086419753085,'accuracy': 0.19753086419753085, 'pred': 1, 'score': 0.19753086419753085, 'pos_count': 32.0,'neg_count': 130.0},
        #testset:{0: 1414, 1: 492, 'recall': 0.23676612127045235, 'precision': 0.25813221406086045,'accuracy': 0.25813221406086045, 'pred': 1, 'score': 0.25813221406086045, 'pos_count': 492.0,'neg_count': 1414.0}])])
        paths, rule_links = booster.get_leaf_path(i, gen_rule_link=True, with_score=False, missing_value=True)
        paths = paths[0]
        rule_links = rule_links[0]
        for leaf_id in top_leaf_ids:
            rule = paths[leaf_id] + ' then {}'.format(round(top_rules_info[leaf_id][0]['score'], 6))
            rule_list.append({
                'rule_info': top_rules_info[leaf_id],
                'rule_link': rule_links[leaf_id],
                'rule': rule,
                'tree_id': i,
                'rule_id': '{}_{}'.format(i, leaf_id)
            })

        evaluation_result_list = []
        train_set, size, index = build_new_dataset(train_set, pred_leaf_list[0], top_leaf_ids)
        train_index.append(index)
        global_min_train_set_size = global_sync_up_by_min(size)
        if global_min_train_set_size == 0:
            logging.warning('iter {}, local dataset size {}, global min dataset size is 0, stop training!'.format(i, size))
            break
        else:
            global_train_set_size = global_sync_up_by_sum(size)
            logging.warning('iter {}, remaining {} samples to train in local worker, {} samples overall.'.format(i, size, int(global_train_set_size)))


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
    return booster, rule_list ,train_index


def compute_top_n_leaf_info(pred_leaf_list, datasets_stats, labels, top_n, rule_pos_rate_threshold=None, classifier_threshold=0.5):
    leaf_info = {}
    #print(len(pred_leaf_list))
    for i in range(len(pred_leaf_list)):
        #print(datasets_stats[i])
        leaf_info_temp = compute_leaf_info(pred_leaf_list[i], labels[i], datasets_stats[i], classifier_threshold)
        for id in leaf_info_temp:
            if id not in leaf_info:
                leaf_info[id] = []
            leaf_info[id].append(leaf_info_temp[id])
    #leaf_info[6] = (6:{训练{0:xx 1:xx}/测试})
    # leaf_info = compute_leaf_info(preds_leaf, labels)
    top_rules = OrderedDict()
    eval_data_idx = 0
    if len(pred_leaf_list) >= 2:
        eval_data_idx = 1
    #top_rules从leaf_info里取值
    #print(eval_data_idx) = 1
    if rule_pos_rate_threshold and top_n >= 0:
        for item in leaf_info.items():
            if item[1][eval_data_idx]['precision'] >= rule_pos_rate_threshold:
                top_rules[item[0]] = item[1]
    else:
        if top_n is None or top_n < 0:
            top_n = len(leaf_info)
        leaf_info_list = sorted(leaf_info.items(), key=lambda item: item[1][eval_data_idx]['precision'], reverse=True)
        #print(len(leaf_info_list))
        for i in range(min(top_n,len(leaf_info_list))):
            top_rules[leaf_info_list[i][0]] = leaf_info_list[i][1]
    logging.warning('select {} rules this round.'.format(len(top_rules)))
    for rule_id in top_rules:
        #rule_id 1 4 0 2 6 5 3 7
        for item in top_rules[rule_id]:
            #输出
            logging.warning('rule_id: {}, prec: {}, recall: {}, pos_count: {}, neg_count: {}'.format(
                rule_id, item['precision'], item['recall'], item['pos_count'], item['neg_count']
            ))

    return top_rules


def compute_leaf_info(preds_leaf, labels, stats, threshold):
    leaf_info = {}
    #preds_leaf [6 4 3 1 3 6 1 6 1 1]
    #labels [ 1.  1.  0. ...,  0.  0.  0.]
    for i in range(len(preds_leaf)):
        leaf_id = preds_leaf[i]
        #leaf_id不在info里 初始化 否则统计 leaf_id下正负样本数目 例如树6下正负样本个数(6:[{0:xx 1:xx}])
        if leaf_id not in leaf_info:
            leaf_info[leaf_id] = {0: 0, 1: 0}
        leaf_info[leaf_id][labels[i]] += 1
    #出现过的leaf信息
    for leaf_id in leaf_info:
        global_count_0 = global_sync_up_by_sum(leaf_info[leaf_id][0])
        global_count_1 = global_sync_up_by_sum(leaf_info[leaf_id][1])
        #global_count_0数字
        positive_rate = float(global_count_1) / (global_count_1 + global_count_0)
        if positive_rate >= threshold:
            global_leaf_pred_class = 0
        else:
            global_leaf_pred_class = 1
        global_accuracy = float(global_count_0) / (global_count_1 + global_count_0) if global_leaf_pred_class == 0 \
            else float(global_count_1) / (global_count_1 + global_count_0)
        #正负样例在此输出
        leaf_info[leaf_id]['recall'] = float(global_count_1) / stats[1]
        leaf_info[leaf_id]['precision'] = positive_rate
        leaf_info[leaf_id]['accuracy'] = global_accuracy
        leaf_info[leaf_id]['pred'] = global_leaf_pred_class
        leaf_info[leaf_id]['score'] = positive_rate
        leaf_info[leaf_id]['pos_count'] = global_count_1
        leaf_info[leaf_id]['neg_count'] = global_count_0
        alpha = 1
        leaf_info[leaf_id]['F'] = (alpha**2+1)*positive_rate*float(global_count_1) / stats[1]/(positive_rate + float(global_count_1) / stats[1] + 1e-10)/alpha**2
    return leaf_info
def build_new_dataset(origin_dataset, pred_leaf, drop_leaf_id):
    used_indices = []
    drop_indices = []
    index = []
    #pred_leaf叶子节点的id不在被选出来的id里 记录i到used_indices = []
    for i in range(len(pred_leaf)):
        if pred_leaf[i] not in drop_leaf_id:
            used_indices.append(i)
        else:
            drop_indices.append(i)
            index.append([i,list(drop_leaf_id).index(pred_leaf[i])])
    # print('drop_leaf_id', drop_leaf_id)
    # print('drop_leaf_id', list(drop_leaf_id))
    # print('pred_leaf', pred_leaf)
    # print('len_pred_leaf', len(pred_leaf))
    # print('used_indices',used_indices)
    # print('drop_indices', drop_indices)

    if len(used_indices) == 0:
        warnings.warn('used_indices is empty, all train data has been covered!')
        return None, 0, index
    new_dataset = origin_dataset.subset(used_indices)
    # new_dataset.data = origin_dataset.data
    #print(used_indices)
    #print(len(used_indices))

    return new_dataset, len(used_indices),index

def build_new_dataset_v2(origin_dataset, pred_leaf, drop_leaf_id):
    used_indices = []
    new_label = []
    for i in range(len(pred_leaf)):
        if pred_leaf[i] not in drop_leaf_id:
            used_indices.append(i)
            new_label.append(origin_dataset.label[i])
    if len(used_indices) == 0:
        warnings.warn('used_indices is empty, all data has been covered!')
        return None, 0
    if isinstance(origin_dataset.data, np.ndarray):
        new_data = origin_dataset.data[used_indices]
    else:
        new_data = origin_dataset.data.iloc[used_indices].reset_index(drop=True)
    new_dataset = Dataset(new_data,
                          label=new_label,
                          reference=origin_dataset,
                          weight=origin_dataset.weight,
                          group=origin_dataset.group,
                          init_score=origin_dataset.init_score,
                          feature_name=origin_dataset.feature_name,
                          categorical_feature=origin_dataset.categorical_feature,
                          params=origin_dataset.params,
                          free_raw_data=origin_dataset.free_raw_data)
    origin_dataset._free_handle()
    return new_dataset, len(used_indices)
