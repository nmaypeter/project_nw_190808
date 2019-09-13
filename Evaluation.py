from Initialization import *
from random import choice
import random
import time
import os


class Evaluation:
    def __init__(self, graph_dict, prod_list, wallet_dict):
        ### graph_dict: (dict) the graph
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = prod_list
        self.num_product = len(prod_list)
        self.wallet_dict = wallet_dict

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        a_n_set = [s_total_set.copy() for _ in range(self.num_product)]
        a_n_sequence, a_n_sequence2 = [(k, s) for k in range(self.num_product) for s in s_set[k]], []
        pro_k_list = [0.0 for _ in range(self.num_product)]
        wallet_dict = self.wallet_dict.copy()

        while a_n_sequence:
            k_prod, i_node = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))
            benefit, price = self.product_list[k_prod][0], self.product_list[k_prod][2]

            for ii_node in self.graph_dict[i_node]:
                if random.random() > self.graph_dict[i_node][ii_node]:
                    continue

                # -- notice: seed cannot use other product --
                if ii_node in a_n_set[k_prod]:
                    continue
                if wallet_dict[ii_node] < price:
                    continue

                # -- purchasing --
                a_n_set[k_prod].add(ii_node)
                wallet_dict[ii_node] -= price
                pro_k_list[k_prod] += benefit

                # -- passing the information --
                if ii_node in self.graph_dict:
                    a_n_sequence2.append((k_prod, ii_node))

            if not a_n_sequence:
                a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        pro_k_list = [round(pro_k, 4) for pro_k in pro_k_list]
        pnn_k_list = [len(a_n_set[k]) - len(s_set[k]) for k in range(self.num_product)]

        return pro_k_list, pnn_k_list


class EvaluationM:
    def __init__(self, model_name, dataset_name, product_name, seed_cost_option, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.seed_cost_option = seed_cost_option
        self.cascade_model = cascade_model
        self.eva_monte_carlo = 100

    def evaluate(self, bi, wallet_distribution_type, sample_seed_set, ss_time):
        eva_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name, wallet_distribution_type)

        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()[0]
        num_product = len(product_list)
        wallet_dict = ini.constructWalletDict()
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))
        total_budget = round(total_cost / 2 ** bi, 4)

        eva = Evaluation(graph_dict, product_list, wallet_dict)
        print('@ ' + self.model_name + ' evaluation @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
              ', seed_cost_option = ' + self.seed_cost_option + ', wd = ' + wallet_distribution_type)
        sample_pro_k, sample_pnn_k = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]

        for _ in range(self.eva_monte_carlo):
            pro_k_list, pnn_k_list = eva.getSeedSetProfit(sample_seed_set)
            sample_pro_k = [(pro_k + sample_pro_k) for pro_k, sample_pro_k in zip(pro_k_list, sample_pro_k)]
            sample_pnn_k = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k)]
        sample_pro_k = [round(sample_pro_k / self.eva_monte_carlo, 4) for sample_pro_k in sample_pro_k]
        sample_pnn_k = [round(sample_pnn_k / self.eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k]
        sample_bud_k = [round(sum(seed_cost_dict[sample_seed_set.index(sample_bud_k)][i] for i in sample_bud_k), 4) for sample_bud_k in sample_seed_set]
        sample_sn_k = [len(sample_sn_k) for sample_sn_k in sample_seed_set]
        sample_pro = round(sum(sample_pro_k), 4)
        sample_bud = round(sum(sample_bud_k), 4)

        result = [sample_pro, sample_bud, sample_sn_k, sample_pnn_k, sample_pro_k, sample_bud_k, sample_seed_set]
        print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
        print(result)
        print('------------------------------------------')

        path = 'result/' + self.model_name + '_' + wallet_distribution_type
        if not os.path.isdir(path):
            os.mkdir(path)
        fw = open(path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '_' + self.seed_cost_option + '_bi' + str(bi) + '.txt', 'w')
        fw.write(self.model_name + ', ' + wallet_distribution_type + ', ' + self.dataset_name + '_' + self.cascade_model + ', ' + self.product_name +
                 ', seed_cost_option = ' + self.seed_cost_option + '\n\n' +
                 'total_budget = ' + str(total_budget) + '\n' +
                 'profit = ' + str(sample_pro) + '\n' +
                 'budget = ' + str(sample_bud) + '\n' +
                 'time = ' + str(ss_time) + '\n')
        fw.write('\nprofit_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_pro_k[kk]))
        fw.write('\nbudget_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_bud_k[kk]))
        fw.write('\nseed_number =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_sn_k[kk]))
        fw.write('\ncustomer_number =')
        for kk in range(num_product):
            fw.write(' ' + str(sample_pnn_k[kk]))
        fw.write('\n\n')

        for r in result:
            # -- pro, bud, sn_k, pnn_k, pro_k, bud_k, seed_set --
            fw.write(str(r) + '\t')
        fw.close()