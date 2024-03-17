import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

dataset_name = 'Ciao'

dataset_path = 'dataset/' + dataset_name + '/'


def process_dataset():
    with open(dataset_path + 'dataset_filter5.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(dataset_path + 'list_filter5.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)

    train_user_set = set([u for u, i, r in train_set if r != 0])
    train_item_set = set([i for u, i, r in train_set if r != 0])
    new_valid_set = []
    for u, i, r in valid_set:
        if u in train_user_set and i in train_item_set and r != 0:
            new_valid_set.append((u, i, r))

    new_test_set = []
    for u, i, r in test_set:
        if u in train_user_set and i in train_item_set and r != 0:
            new_test_set.append((u, i, r))

    print('valid set and filtered valid set len: ', len(valid_set), len(new_valid_set))
    print('test set and filtered test set len: ', len(test_set), len(new_test_set))

    # 先统计training数据集中user的点击个数
    training_rating_dict = defaultdict(list)
    for u, i, r in train_set:
        if r == 0:
            continue
        training_rating_dict[u].append((i, r))

    n_20, n_40, n_80, n_160, n_320 = set(), set(), set(), set(), set()
    cold_rating_list_20, cold_rating_list_40, cold_rating_list_80, cold_rating_list_160, cold_rating_list_320 = [], [], [], [], []
    for u, i, r in new_test_set:  # 原本统计test，现在查看training set
        if len(training_rating_dict[u]) < 20:
            cold_rating_list_20.append((u, i, r))
            n_20.add(u)
        elif len(training_rating_dict[u]) >= 20 and len(training_rating_dict[u]) < 40:
            cold_rating_list_40.append((u, i, r))
            n_40.add(u)
        elif len(training_rating_dict[u]) >= 40 and len(training_rating_dict[u]) < 80:
            cold_rating_list_80.append((u, i, r))
            n_80.add(u)
        elif len(training_rating_dict[u]) >= 80 and len(training_rating_dict[u]) < 160:
            cold_rating_list_160.append((u, i, r))
            n_160.add(u)
        else:
            cold_rating_list_320.append((u, i, r))
            n_320.add(u)

    print('test len: ', len(new_test_set), 'cold rating_list lenth: ', len(cold_rating_list_20),
          len(cold_rating_list_40), len(cold_rating_list_80), len(cold_rating_list_160), len(cold_rating_list_320))
    print('cold rating list of user distribution: ', len(n_20), len(n_40), len(n_80), len(n_160), len(n_320))

    # 统计user的social relation个数
    cold_start_user_list_0, cold_start_user_list_5, cold_start_user_list_10, cold_start_user_list_20, cold_start_user_list_40, cold_start_user_list_80 = [], [], [], [], [], []
    for u in range(1, user_count + 1):
        if u_users_list[u] == [0]:
            cold_start_user_list_0.append(u)
        elif u_users_list[u] != [0] and len(u_users_list[u]) < 5:
            cold_start_user_list_5.append(u)
        elif u_users_list[u] != [0] and len(u_users_list[u]) >= 5 and len(u_users_list[u]) < 10:
            cold_start_user_list_10.append(u)
        elif u_users_list[u] != [0] and len(u_users_list[u]) >= 10 and len(u_users_list[u]) < 20:
            cold_start_user_list_20.append(u)
        elif u_users_list[u] != [0] and len(u_users_list[u]) >= 20 and len(u_users_list[u]) < 40:
            cold_start_user_list_40.append(u)
        else:
            cold_start_user_list_80.append(u)

    cold_social_list_0, cold_social_list_5, cold_social_list_10, cold_social_list_20, cold_social_list_40, cold_social_list_80 = [], [], [], [], [], []
    for u, i, r in new_test_set:
        if u in set(cold_start_user_list_0):
            cold_social_list_0.append((u, i, r))
        elif u in set(cold_start_user_list_5):
            cold_social_list_5.append((u, i, r))
        elif u in set(cold_start_user_list_10):
            cold_social_list_10.append((u, i, r))
        elif u in set(cold_start_user_list_20):
            cold_social_list_20.append((u, i, r))
        elif u in set(cold_start_user_list_40):
            cold_social_list_40.append((u, i, r))
        else:
            cold_social_list_80.append((u, i, r))

    print('cold_start_social_list lenth: ', len(cold_social_list_0), len(cold_social_list_5), len(cold_social_list_10),
          len(cold_social_list_20), len(cold_social_list_40), len(cold_social_list_80))
    print('cold social list of user distribution: ', len(cold_start_user_list_0), len(cold_start_user_list_5),
          len(cold_start_user_list_10), len(cold_start_user_list_20), len(cold_start_user_list_40),
          len(cold_start_user_list_80))

    with open('cold_start_rating_%s.pkl' % dataset_name, 'wb') as f:
        pickle.dump(cold_rating_list_20, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_rating_list_40, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_rating_list_80, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_rating_list_160, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_rating_list_320, f, pickle.HIGHEST_PROTOCOL)

    with open('cold_start_social_%s.pkl' % dataset_name, 'wb') as f:
        pickle.dump(cold_social_list_0, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_social_list_5, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_social_list_10, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_social_list_20, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_social_list_40, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cold_social_list_80, f, pickle.HIGHEST_PROTOCOL)


def s4rec_cold_social_analysis(filename_deep, filename_wide):
    with open('%s.txt' % filename_deep, 'r') as f:
        for line in f:
            loss_list = json.loads(line.strip())

    deep_loss_dict = dict()
    for u, i, r, rp in loss_list:
        deep_loss_dict[str(u) + '-' + str(i)] = float(rp)

    wide_loss_dict = dict()
    with open('%s' % filename_wide, 'rb') as f:
        for line in f:
            data = line.decode().strip().split(',')
            wide_loss_dict[data[0] + '-' + data[1]] = float(data[2])

    with open('cold_start_social_%s.pkl' % dataset_name, 'rb') as f:
        cold_rating_list_0 = pickle.load(f)
        cold_rating_list_5 = pickle.load(f)
        cold_rating_list_10 = pickle.load(f)
        cold_rating_list_20 = pickle.load(f)
        cold_rating_list_40 = pickle.load(f)
        cold_rating_list_80 = pickle.load(f)

    cold_rating_0_dict, cold_rating_5_dict, cold_rating_10_dict, cold_rating_20_dict, cold_rating_40_dict, cold_rating_80_dict = dict(), dict(), dict(), dict(), dict(), dict()
    for u, i, r in cold_rating_list_0:
        cold_rating_0_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_5:
        cold_rating_5_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_10:
        cold_rating_10_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_20:
        cold_rating_20_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_40:
        cold_rating_40_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_80:
        cold_rating_80_dict[str(u) + '-' + str(i)] = r

    loss_0, loss_5, loss_10, loss_20, loss_40, loss_80 = [], [], [], [], [], []

    weight = 0.55

    for key, r in deep_loss_dict.items():
        rp = weight * r + (1 - weight) * wide_loss_dict[key]
        if key in cold_rating_0_dict:
            loss_0.append(abs(rp - cold_rating_0_dict[key]))
        elif key in cold_rating_5_dict:
            loss_5.append(abs(rp - cold_rating_5_dict[key]))
        elif key in cold_rating_10_dict:
            loss_10.append(abs(rp - cold_rating_10_dict[key]))
        elif key in cold_rating_20_dict:
            loss_20.append(abs(rp - cold_rating_20_dict[key]))
        elif key in cold_rating_40_dict:
            loss_40.append(abs(rp - cold_rating_40_dict[key]))
        else:
            loss_80.append(abs(rp - cold_rating_80_dict[key]))

    print(len(loss_0), len(loss_5), len(loss_10), len(loss_20), len(loss_40), len(loss_80))
    print('\n', 'fuse: ')
    print('cold start rating: ', np.mean(loss_0), np.sqrt(np.mean(np.power(loss_0, 2))))
    print('cold start rating: ', np.mean(loss_5), np.sqrt(np.mean(np.power(loss_5, 2))))
    print('cold start rating: ', np.mean(loss_10), np.sqrt(np.mean(np.power(loss_10, 2))))
    print('cold start rating: ', np.mean(loss_20), np.sqrt(np.mean(np.power(loss_20, 2))))
    print('cold start rating: ', np.mean(loss_40), np.sqrt(np.mean(np.power(loss_40, 2))))
    print('cold start rating: ', np.mean(loss_80), np.sqrt(np.mean(np.power(loss_80, 2))))


#######################################################################################################
#######################################################################################################
#######################################################################################################


def s4rec_cold_rating_analysis(filename_deep, filename_wide):
    with open('%s.txt' % filename_deep, 'r') as f:
        for line in f:
            loss_list = json.loads(line.strip())
    deep_loss_dict = dict()
    for u, i, r, rp in loss_list:
        deep_loss_dict[str(u) + '-' + str(i)] = float(rp)

    wide_loss_dict = dict()
    with open('%s' % filename_wide, 'rb') as f:
        for line in f:
            data = line.decode().strip().split(',')
            wide_loss_dict[data[0] + '-' + data[1]] = float(data[2])

    print(len(deep_loss_dict), len(wide_loss_dict))

    with open('cold_start_rating_%s.pkl' % dataset_name, 'rb') as f:
        cold_rating_list_20 = pickle.load(f)
        cold_rating_list_40 = pickle.load(f)
        cold_rating_list_80 = pickle.load(f)
        cold_rating_list_160 = pickle.load(f)
        cold_rating_list_320 = pickle.load(f)

    cold_rating_20_dict, cold_rating_40_dict, cold_rating_80_dict, cold_rating_160_dict, cold_rating_320_dict, = dict(), dict(), dict(), dict(), dict()
    total_rating_dict = dict()
    for u, i, r in cold_rating_list_20:
        cold_rating_20_dict[str(u) + '-' + str(i)] = r
        total_rating_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_40:
        cold_rating_40_dict[str(u) + '-' + str(i)] = r
        total_rating_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_80:
        cold_rating_80_dict[str(u) + '-' + str(i)] = r
        total_rating_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_160:
        cold_rating_160_dict[str(u) + '-' + str(i)] = r
        total_rating_dict[str(u) + '-' + str(i)] = r
    for u, i, r in cold_rating_list_320:
        cold_rating_320_dict[str(u) + '-' + str(i)] = r
        total_rating_dict[str(u) + '-' + str(i)] = r

    loss_20, loss_40, loss_80, loss_160, loss_320 = [], [], [], [], []

    weight = 0.55
    fuse_rating_dict = dict()

    for key, r in deep_loss_dict.items():
        if key in cold_rating_20_dict:
            rp = weight * r + (1 - weight) * wide_loss_dict[key]
            loss_20.append(abs(rp - cold_rating_20_dict[key]))
            fuse_rating_dict[key] = rp
        elif key in cold_rating_40_dict:
            rp = weight * r + (1 - weight) * wide_loss_dict[key]
            loss_40.append(abs(rp - cold_rating_40_dict[key]))
            fuse_rating_dict[key] = rp
        elif key in cold_rating_80_dict:
            rp = weight * r + (1 - weight) * wide_loss_dict[key]
            loss_80.append(abs(rp - cold_rating_80_dict[key]))
            fuse_rating_dict[key] = rp
        elif key in cold_rating_160_dict:
            rp = weight * r + (1 - weight) * wide_loss_dict[key]
            loss_160.append(abs(rp - cold_rating_160_dict[key]))
            fuse_rating_dict[key] = rp
        else:
            rp = weight * r + (1 - weight) * wide_loss_dict[key]
            loss_320.append(abs(rp - cold_rating_320_dict[key]))
            fuse_rating_dict[key] = rp

    print(len(loss_20), len(loss_40), len(loss_80), len(loss_160), len(loss_320))
    print('cold start rating: ', np.mean(loss_20), np.sqrt(np.mean(np.power(loss_20, 2))))
    print('cold start rating: ', np.mean(loss_40), np.sqrt(np.mean(np.power(loss_40, 2))))
    print('cold start rating: ', np.mean(loss_80), np.sqrt(np.mean(np.power(loss_80, 2))))
    print('cold start rating: ', np.mean(loss_160), np.sqrt(np.mean(np.power(loss_160, 2))))
    print('cold start rating: ', np.mean(loss_320), np.sqrt(np.mean(np.power(loss_320, 2))))

    x = []
    x.extend(loss_20)
    x.extend(loss_40)
    x.extend(loss_80)
    x.extend(loss_160)
    x.extend(loss_320)
    print('total : ', np.mean(x), np.sqrt(np.mean(np.power(x, 2))))


def evaluation_deep(filename_deep):
    print('-----------------开始评估deep指标-------------------------')
    # 步骤1: 从字典转换为每个用户的评分列表与实际评分列表
    user_recommended_rates_deep = defaultdict(list)
    user_actual_rates = defaultdict(list)

    # 打开深度图模型的预测结果
    with open('%s.txt' % filename_deep, 'r') as f:
        for line in f:
            loss_list = json.loads(line.strip())
    # 加载深度图模型的预测结果
    for u, i, r, rp in loss_list:  # user item rate predict-rate
        user_recommended_rates_deep[u].append((i, rp))
        user_actual_rates[u].append(i)  # 只需要item_id
    # 为了确保推荐列表是根据评分排序的
    for user in user_recommended_rates_deep:
        user_recommended_rates_deep[user] = sorted(user_recommended_rates_deep[user], key=lambda x: x[1],
                                                   reverse=True)

        # 步骤3: 计算指标
    # topk
    ks = [5, 10, 20, 30]
    # 要计算的指标
    metrics = {
        'hit_ratio': {k: [] for k in ks},
        'recall': {k: [] for k in ks},
        'precision': {k: [] for k in ks},
        'ndcg': {k: [] for k in ks},
        'mrr': {k: [] for k in ks}
    }
    for user in user_actual_rates:
        recommended_items = [item for item, rate in user_recommended_rates_deep[user]]
        actual_items = user_actual_rates[user]
        for k in ks:
            metrics['hit_ratio'][k].append(hit_ratio_at_k(recommended_items, actual_items, k))
            metrics['recall'][k].append(recall_at_k(recommended_items, actual_items, k))
            metrics['precision'][k].append(precision_at_k(recommended_items, actual_items, k))
            metrics['ndcg'][k].append(ndcg_at_k(recommended_items, actual_items, k))
            metrics['mrr'][k].append(mrr_at_k(recommended_items, actual_items, k))

    # 创建一个空的DataFrame用于存储结果
    columns = ['Top-K', 'Hit Ratio', 'Recall', 'Precision', 'NDCG', 'MRR']
    results_df = pd.DataFrame(columns=columns)
    # 步骤4: 打印指标
    for k in ks:
        row = {
            'Top-K': k,
            'Hit Ratio': np.mean(metrics['hit_ratio'][k]),
            'Recall': np.mean(metrics['recall'][k]),
            'Precision': np.mean(metrics['precision'][k]),
            'NDCG': np.mean(metrics['ndcg'][k]),
            'MRR': np.mean(metrics['mrr'][k])
        }
        results_df = results_df._append(row, ignore_index=True)
        print(f'Top-{k}')
        print(f"Hit Ratio: {np.mean(metrics['hit_ratio'][k]):.4f}")
        print(f"Recall: {np.mean(metrics['recall'][k]):.4f}")
        print(f"Precision: {np.mean(metrics['precision'][k]):.4f}")
        print(f"NDCG: {np.mean(metrics['ndcg'][k]):.4f}")
        print(f"MRR: {np.mean(metrics['mrr'][k]):.4f}")
        print()
    # 将DataFrame保存为CSV文件，这里可以添加数据集名称和参数到文件名中
    dataset_name = 'Ciao'
    # model_parameters = 'params1'  # 描述这次实验参数的字符串
    csv_filename = f'{dataset_name}_metrics_deep_v0.csv'
    results_df.to_csv(csv_filename, index=False)
    print('-----------------结束评估deep指标-------------------------')


def evaluation_s4rec(filename_deep, filename_wide):
    print('-----------------开始评估s4rec指标-------------------------')
    # 打开深度图模型的预测结果
    with open('%s.txt' % filename_deep, 'r') as f:
        for line in f:
            loss_list = json.loads(line.strip())
    deep_loss_dict = dict()
    # 加载深度图模型的预测结果
    for u, i, r, rp in loss_list:  # user item rate predict-rate
        deep_loss_dict[str(u) + '-' + str(i)] = float(rp)  # {u-i : predict-rate}

    # 加载广度图模型的预测结果
    wide_loss_dict = dict()
    with open('%s' % filename_wide, 'rb') as f:
        for line in f:
            data = line.decode().strip().split(',')
            wide_loss_dict[data[0] + '-' + data[1]] = float(data[2])  # {u-i : predict-rate}

    print(len(deep_loss_dict), len(wide_loss_dict))

    # 步骤1: 从字典转换为每个用户的评分列表与实际评分列表
    user_recommended_rates_s4rec = defaultdict(list)
    user_actual_rates = defaultdict(list)

    for key, r in deep_loss_dict.items():
        # try:
        #     rp = weight * r + (1 - weight) * wide_loss_dict[key]
        # except KeyError:
        #     key_error_count += 1
        #     continue
        rp = weight * r + (1 - weight) * wide_loss_dict[key]
        user, item = key.split('-')
        # 推荐物品评分列表
        user_recommended_rates_s4rec[user].append((item, rp))
        # 实际物品评分列表
        user_actual_rates[user].append(item)  # 只需要item_id
    # print(f"Number of skipped entries due to KeyError: {key_error_count}")

    # 为了确保推荐列表是根据评分排序的
    for user in user_recommended_rates_s4rec:
        user_recommended_rates_s4rec[user] = sorted(user_recommended_rates_s4rec[user], key=lambda x: x[1],
                                                    reverse=True)

    # 步骤3: 计算指标
    ks = [5, 10, 20, 30]
    metrics = {
        'hit_ratio': {k: [] for k in ks},
        'recall': {k: [] for k in ks},
        'precision': {k: [] for k in ks},
        'ndcg': {k: [] for k in ks},
        'mrr': {k: [] for k in ks}
    }

    for user in user_actual_rates:
        recommended_items = [item for item, rate in user_recommended_rates_s4rec[user]]
        actual_items = user_actual_rates[user]
        for k in ks:
            metrics['hit_ratio'][k].append(hit_ratio_at_k(recommended_items, actual_items, k))
            metrics['recall'][k].append(recall_at_k(recommended_items, actual_items, k))
            metrics['precision'][k].append(precision_at_k(recommended_items, actual_items, k))
            metrics['ndcg'][k].append(ndcg_at_k(recommended_items, actual_items, k))
            metrics['mrr'][k].append(mrr_at_k(recommended_items, actual_items, k))

    # 步骤4: 打印指标
    # 创建一个空的DataFrame用于存储结果
    columns = ['Top-K', 'Hit Ratio', 'Recall', 'Precision', 'NDCG', 'MRR']
    results_df = pd.DataFrame(columns=columns)
    for k in ks:
        row = {
            'Top-K': k,
            'Hit Ratio': np.mean(metrics['hit_ratio'][k]),
            'Recall': np.mean(metrics['recall'][k]),
            'Precision': np.mean(metrics['precision'][k]),
            'NDCG': np.mean(metrics['ndcg'][k]),
            'MRR': np.mean(metrics['mrr'][k])
        }
        results_df = results_df._append(row, ignore_index=True)
        print(f'Top-{k}')
        print(f"Hit Ratio: {np.mean(metrics['hit_ratio'][k]):.4f}")
        print(f"Recall: {np.mean(metrics['recall'][k]):.4f}")
        print(f"Precision: {np.mean(metrics['precision'][k]):.4f}")
        print(f"NDCG: {np.mean(metrics['ndcg'][k]):.4f}")
        print(f"MRR: {np.mean(metrics['mrr'][k]):.4f}")
        print()
    # 将DataFrame保存为CSV文件，这里可以添加数据集名称和参数到文件名中
    dataset_name = 'Ciao'
    # model_parameters = 'params1'  # 描述这次实验参数的字符串
    csv_filename = f'{dataset_name}_metrics_s4rec_v0.csv'
    results_df.to_csv(csv_filename, index=False)

    print('-----------------结束评估s4rec指标-------------------------')


def hit_ratio_at_k(recommended_list, actual_list, k):
    return int(len(set(recommended_list[:k]) & set(actual_list)) > 0)


def recall_at_k(recommended_list, actual_list, k):
    return len(set(recommended_list[:k]) & set(actual_list)) / len(actual_list)


def precision_at_k(recommended_list, actual_list, k):
    return len(set(recommended_list[:k]) & set(actual_list)) / k


def ndcg_at_k(recommended_list, actual_list, k):
    k = min(k, len(recommended_list))
    idcg = sum((1.0 / np.log2(i + 2) for i in range(min(len(actual_list), k))))
    dcg = sum((1.0 / np.log2(i + 2) if recommended_list[i] in actual_list else 0.0 for i in range(k)))
    return dcg / idcg if idcg > 0 else 0


def mrr_at_k(recommended_list, actual_list, k):
    for rank, item in enumerate(recommended_list[:k]):
        if item in actual_list:
            return 1 / (rank + 1)
    return 0.0


if __name__ == '__main__':
    # process_dataset()
    # s4rec_cold_rating_analysis('results/%s/test_best_predict_list' %dataset_name, 'wide_ciao/trustsvd')
    # s4rec_cold_social_analysis('results/%s/test_best_predict_list' %dataset_name, 'wide_ciao/trustsvd')
    evaluation_deep('results/%s/test_best_predict_list' % dataset_name, )
    # evaluation_s4rec('results/%s/test_best_predict_list' %dataset_name, 'wide_ciao/trustsvd')

