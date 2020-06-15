#train data during backfitting
import lightgbm as lgb
from lightgbm import Dataset
from odps import ODPS


def pre_train(df_train_x,train_y,df_Validation_x,Validation_y):
    class ODPSWriter(object):
        def __init__(self):
            self._o = ODPS('*****',
                           '*****',
                           'sre_mpi_algo_dev',
                           'http://service-corp.odps.aliyun-inc.com/api')

        def open_table(self, table_name, out_cols, drop_if_exists=True):
            if drop_if_exists and self._o.exist_table(table_name):
                table = self._o.get_table(table_name)
                table.drop()
            tmp_table = self._o.create_table(
                table_name, out_cols, if_not_exists=False
            )
            self._odps_table = self._o.get_table(table_name)
            self._writer = self._odps_table.open_writer()

        # batchWrite Records
        def batch_write(self, records):
            self._writer.write(records)

        # Flush & Close table
        def close(self):
            self._writer.close()

        # get table record
        def gen_record(self):
            return self._odps_table.new_record()

    def upload_rule_list(rule_list, table_name, lifecycle=28):
        if table_name is None:
            return
        schema = 'treeid bigint, ruleid string, rulelink string, sql_rules string, ' \
                 'train_accuracy double, valid_accuracy double, train_precision double, valid_precision double, ' \
                 'train_recall double, valid_recall double, train_pos_count bigint, valid_pos_count bigint, ' \
                 'train_neg_count bigint, valid_neg_count bigint'
        rule_records = []
        writer = ODPSWriter()
        writer.open_table(table_name, schema)
        for rule in rule_list:
            record = writer.gen_record()
            record['treeid'] = rule['tree_id']
            record['ruleid'] = rule['rule_id']
            record['rulelink'] = rule['rule_link']
            record['sql_rules'] = rule['rule']
            record['train_accuracy'] = rule['rule_info'][0]['accuracy']
            record['valid_accuracy'] = rule['rule_info'][1]['accuracy']
            record['train_precision'] = rule['rule_info'][0]['precision']
            record['valid_precision'] = rule['rule_info'][1]['precision']
            record['train_recall'] = rule['rule_info'][0]['recall']
            record['valid_recall'] = rule['rule_info'][1]['recall']
            record['train_pos_count'] = rule['rule_info'][0]['pos_count']
            record['valid_pos_count'] = rule['rule_info'][1]['pos_count']
            record['train_neg_count'] = rule['rule_info'][0]['neg_count']
            record['valid_neg_count'] = rule['rule_info'][1]['neg_count']
            rule_records.append(record)
        writer.batch_write(rule_records)
        writer.close()
    train_dataset = Dataset(df_train_x, label=train_y, free_raw_data=False)
    valid_dataset = Dataset(df_Validation_x, label=Validation_y, free_raw_data=False)
    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'num_iteration': 20,
        'learning_rate': 0.2,
        'num_leaves': 15,
        'max_depth': 5,
        # 'verbose': -1,
        'seed': 7,
        'mode': 'sequential_covering',
        'min_data_in_leaf': 50,

        'lambda_l2': 1.0,
        'lambda_l1': 1.0,
        'max_bin': 255,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 1,
        'min_data_in_bin': 1,
        'rule_top_n_each_tree': 5,
        'num_stop_threshold': 10000,
        # 'rule_pos_rate_threshold': 0.5,
        # 'categorical_feature': '0,1'
    }
    booster, rules = lgb.train(params, train_dataset, valid_sets=valid_dataset)
    return rules
