from SeedSelection import *
from Evaluation import *
import time
import copy


def generateHeapOrder(heap_o, model_name, dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type):
    heap_des = sorted(heap_o, reverse=True)
    wd_seq = [wallet_distribution_type] if wallet_distribution_type else ['m50e25', 'm99e96']
    ini = Initialization(dataset_name, product_name, wallet_distribution_type)
    graph_dict = ini.constructGraphDict(cascade_model)
    product_list = ini.constructProductList()[0]
    num_product = len(product_list)

    order_dict = {(k, i): ('', '') for k in range(num_product) for i in graph_dict}
    now_order, last_mg = 0, -1
    for item in heap_des:
        p, k, i = item[0], item[1], item[2]
        now_order += 1 if p != last_mg else 0
        order_dict[(k, i)] = (p, now_order)
        last_mg = p

    for wd in wd_seq:
        path = 'heap_order/' + model_name + '_' + wd
        if not os.path.isdir(path):
            os.mkdir(path)
        fw = open(path + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '_' + seed_cost_option + '_ds' * diff_seed_option + '.txt', 'w')
        for k in range(num_product):
            for i in graph_dict:
                index = (k, i)
                mg, order = order_dict[index]
                fw.write(str(index) + '\t' + str(mg) + '\t' + str(order) + '\n')
        fw.close()


class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type=''):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.seed_cost_option = seed_cost_option
        self.diff_seed_option = diff_seed_option
        self.wallet_distribution_type = wallet_distribution_type
        self.wd_seq = ['m50e25', 'm99e96']
        self.budget_iteration = [i for i in range(10, 6, -1)]
        self.monte_carlo = 100

    def model_mioa(self, r_flag, epw_flag=False):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        seed_mioa_dict = [{} for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        if epw_flag:
            mioa_dict = ssmioa_model.updateMIOAEPW(mioa_dict)
        celf_heap = [(safe_div(sum(mioa_dict[k][i][j][0] for j in mioa_dict[k][i]) * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]), seed_cost_dict[k][i] if r_flag else 1.0), k, i, 0)
                     for k in range(num_product) for i in mioa_dict[k]]
        heap.heapify_max(celf_heap)
        generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.seed_cost_option, self.diff_seed_option, self.wallet_distribution_type)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, seed_mioa_dict, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, seed_mioa_dict, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), copy.deepcopy(seed_mioa_dict), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_mioa_dict = updateSeedMIOADict(seed_mioa_dict, mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_k_prod][mep_i_node])
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = round(now_profit + mep_mg * (sc if r_flag else 1.0), 4)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                    if self.diff_seed_option:
                        celf_heap = [celf_item for celf_item in celf_heap if celf_item[2] != mep_i_node]
                        heap.heapify_max(celf_heap)
                else:
                    seed_exp_mioa_dict = updateSeedMIOADict(copy.deepcopy(seed_mioa_dict), mep_k_prod, mep_i_node, seed_set, mioa_dict[mep_k_prod][mep_i_node])
                    expected_inf = calculateExpectedInf(seed_exp_mioa_dict)
                    ep_t = sum(expected_inf[k] * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]) for k in range(num_product))
                    mg_t = round(ep_t - now_profit, 4)
                    if r_flag:
                        mg_t = safe_div(mg_t, sc)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_dag1(self, r_flag, epw_flag=False):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        seed_dag_dict = [{} for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        if epw_flag:
            mioa_dict = ssmioa_model.updateMIOAEPW(mioa_dict)
        celf_heap = [(safe_div(sum(mioa_dict[k][i][j][0] for j in mioa_dict[k][i]) * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]), seed_cost_dict[k][i] if r_flag else 1.0), k, i, 0)
                     for k in range(num_product) for i in mioa_dict[k]]
        heap.heapify_max(celf_heap)
        mep_seed_dag_dict = (celf_heap[0][0], [{} for _ in range(num_product)])
        mep_seed_dag_dict[1][celf_heap[0][1]] = {celf_heap[0][2]: mioa_dict[celf_heap[0][1]][celf_heap[0][2]]}
        generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.seed_cost_option, self.diff_seed_option, self.wallet_distribution_type)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, seed_dag_dict, mep_seed_dag_dict, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, seed_dag_dict, mep_seed_dag_dict, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), copy.deepcopy(seed_dag_dict), copy.deepcopy(mep_seed_dag_dict), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = round(now_profit + mep_mg * (sc if r_flag else 1.0), 4)
                    seed_dag_dict = mep_seed_dag_dict[1]
                    mep_seed_dag_dict = (0.0, [{} for _ in range(num_product)])
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                    if self.diff_seed_option:
                        celf_heap = [celf_item for celf_item in celf_heap if celf_item[2] != mep_i_node]
                        heap.heapify_max(celf_heap)
                else:
                    seed_exp_dag_dict = copy.deepcopy(seed_dag_dict)
                    updateSeedDAGDict(seed_exp_dag_dict, mep_k_prod, mep_i_node)
                    seed_set_t = seed_set[mep_k_prod].copy()
                    seed_set_t.add(mep_i_node)
                    dag_k_dict = ssmioa_model.generateDAG1(seed_set_t)
                    seed_exp_dag_dict[mep_k_prod] = ssmioa_model.generateSeedDAGDict(dag_k_dict, seed_set_t)
                    expected_inf = calculateExpectedInf(seed_exp_dag_dict)
                    ep_t = sum(expected_inf[k] * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]) for k in range(num_product))
                    mg_t = round(ep_t - now_profit, 4)
                    if r_flag:
                        mg_t = safe_div(mg_t, sc)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)
                        if mg_t > mep_seed_dag_dict[0]:
                            mep_seed_dag_dict = (mg_t, seed_exp_dag_dict)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_dag2(self, r_flag, epw_flag=False):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        seed_dag_dict = [{} for _ in range(num_product)]
        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        mioa_dict = ssmioa_model.generateMIOA()
        if epw_flag:
            mioa_dict = ssmioa_model.updateMIOAEPW(mioa_dict)
        celf_heap = [(safe_div(sum(mioa_dict[k][i][j][0] for j in mioa_dict[k][i]) * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]), seed_cost_dict[k][i] if r_flag else 1.0), k, i, 0)
                     for k in range(num_product) for i in mioa_dict[k]]
        heap.heapify_max(celf_heap)
        mep_seed_dag_dict = (celf_heap[0][0], [{} for _ in range(num_product)])
        mep_seed_dag_dict[1][celf_heap[0][1]] = {celf_heap[0][2]: mioa_dict[celf_heap[0][1]][celf_heap[0][2]]}
        generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.seed_cost_option, self.diff_seed_option, self.wallet_distribution_type)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, seed_dag_dict, mep_seed_dag_dict, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, seed_dag_dict, mep_seed_dag_dict, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), copy.deepcopy(seed_dag_dict), copy.deepcopy(mep_seed_dag_dict), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = round(now_profit + mep_mg * (sc if r_flag else 1.0), 4)
                    seed_dag_dict = mep_seed_dag_dict[1]
                    mep_seed_dag_dict = (0.0, [{} for _ in range(num_product)])
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                    if self.diff_seed_option:
                        celf_heap = [celf_item for celf_item in celf_heap if celf_item[2] != mep_i_node]
                        heap.heapify_max(celf_heap)
                else:
                    seed_exp_dag_dict = copy.deepcopy(seed_dag_dict)
                    updateSeedDAGDict(seed_exp_dag_dict, mep_k_prod, mep_i_node)
                    seed_set_t = seed_set[mep_k_prod].copy()
                    seed_set_t.add(mep_i_node)
                    dag_k_dict = ssmioa_model.generateDAG2(seed_set_t, {i: mioa_dict[mep_k_prod][i] for i in seed_set_t})
                    seed_exp_dag_dict[mep_k_prod] = ssmioa_model.generateSeedDAGDict(dag_k_dict, seed_set_t)
                    expected_inf = calculateExpectedInf(seed_exp_dag_dict)
                    ep_t = sum(expected_inf[k] * product_list[k][0] * (1.0 if epw_flag else product_weight_list[k]) for k in range(num_product))
                    mg_t = round(ep_t - now_profit, 4)
                    if r_flag:
                        mg_t = safe_div(mg_t, sc)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)
                        if mg_t > mep_seed_dag_dict[0]:
                            mep_seed_dag_dict = (mg_t, seed_exp_dag_dict)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_ng(self, r_flag):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag=r_flag)
        diff_model = Diffusion(graph_dict, product_list, product_weight_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget, now_profit = 0.0, 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        celf_heap = ssng_model.generateCelfHeap()
        generateHeapOrder(celf_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.seed_cost_option, self.diff_seed_option, self.wallet_distribution_type)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, now_profit, seed_set, celf_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            celf_heap_c = []
            while now_budget < total_budget and celf_heap:
                if round(now_budget + seed_cost_dict[celf_heap[0][1]][celf_heap[0][2]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    celf_heap_c = copy.deepcopy(celf_heap)
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]
                seed_set_length = sum(len(seed_set[k]) for k in range(num_product))

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, now_profit, copy.deepcopy(seed_set), celf_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                if mep_flag == seed_set_length:
                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)
                    now_profit = safe_div(sum([diff_model.getSeedSetProfit(seed_set) for _ in range(self.monte_carlo)]), self.monte_carlo)
                    seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                     str(now_budget) + '\t' + str(now_profit) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                    if self.diff_seed_option:
                        celf_heap = [celf_item for celf_item in celf_heap if celf_item[2] != mep_i_node]
                        heap.heapify_max(celf_heap)
                else:
                    seed_set_t = copy.deepcopy(seed_set)
                    seed_set_t[mep_k_prod].add(mep_i_node)
                    ep_t = safe_div(sum([diff_model.getSeedSetProfit(seed_set_t) for _ in range(self.monte_carlo)]), self.monte_carlo)
                    mg_t = round(ep_t - now_profit, 4)
                    if r_flag:
                        mg_t = safe_div(mg_t, sc)
                    flag_t = seed_set_length

                    if mg_t > 0:
                        celf_item_t = (mg_t, mep_k_prod, mep_i_node, flag_t)
                        heap.heappush_max(celf_heap, celf_item_t)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_hd(self):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(graph_dict, product_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget = 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        degree_heap = sshd_model.generateDegreeHeap()
        generateHeapOrder(degree_heap, self.model_name, self.dataset_name, self.product_name, self.cascade_model, self.seed_cost_option, self.diff_seed_option, self.wallet_distribution_type)

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, seed_set, degree_heap]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, seed_set, degree_heap] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            degree_heap_c = []
            while now_budget < total_budget and degree_heap:
                if round(now_budget + seed_cost_dict[degree_heap[0][1]][degree_heap[0][2]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    degree_heap_c = copy.deepcopy(degree_heap)
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), degree_heap_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                if self.diff_seed_option:
                    degree_heap = [celf_item for celf_item in degree_heap if celf_item[2] != mep_i_node]
                    heap.heapify_max(degree_heap)

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])

    def model_r(self):
        ini = Initialization(self.dataset_name, self.product_name, self.wallet_distribution_type)
        seed_cost_dict = ini.constructSeedCostDict(self.seed_cost_option)
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list, product_weight_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for k in range(num_product) for i in seed_cost_dict[k])

        seed_set_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [-1 for _ in range(len(self.budget_iteration))]
        seed_data_sequence = [-1 for _ in range(len(self.budget_iteration))]
        ssr_model = SeedSelectionRandom(graph_dict, product_list)

        ss_start_time = time.time()
        bud_iteration = self.budget_iteration.copy()
        now_b_iter = bud_iteration.pop(0)
        now_budget = 0.0
        seed_set = [set() for _ in range(num_product)]

        wd_seq = [self.wallet_distribution_type] if self.wallet_distribution_type else self.wd_seq
        random_node_list = ssr_model.generateRandomList()

        ss_acc_time = round(time.time() - ss_start_time, 4)
        temp_sequence = [[ss_acc_time, now_budget, seed_set, random_node_list]]
        temp_seed_data = [['time\tk_prod\ti_node\tnow_budget\tnow_profit\tseed_num\n']]
        while temp_sequence:
            ss_start_time = time.time()
            now_bi_index = self.budget_iteration.index(now_b_iter)
            total_budget = safe_div(total_cost, 2 ** now_b_iter)
            [ss_acc_time, now_budget, seed_set, random_node_list] = temp_sequence.pop()
            seed_data = temp_seed_data.pop()
            print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', seed_cost_option = ' + self.seed_cost_option + ', bud_iter = ' + str(now_b_iter) + ', budget = ' + str(total_budget))

            random_node_list_c = []
            while now_budget < total_budget and random_node_list:
                if round(now_budget + seed_cost_dict[random_node_list[0][0]][random_node_list[0][1]], 4) >= total_budget and bud_iteration and not temp_sequence:
                    random_node_list_c = copy.deepcopy(random_node_list)
                mep_item = random_node_list.pop(0)
                mep_k_prod, mep_i_node = mep_item
                sc = seed_cost_dict[mep_k_prod][mep_i_node]

                if round(now_budget + sc, 4) >= total_budget and bud_iteration and not temp_sequence:
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    now_b_iter = bud_iteration.pop(0)
                    temp_sequence.append([ss_time, now_budget, copy.deepcopy(seed_set), random_node_list_c])
                    temp_seed_data.append(seed_data.copy())

                if round(now_budget + sc, 4) > total_budget:
                    continue

                seed_set[mep_k_prod].add(mep_i_node)
                now_budget = round(now_budget + sc, 4)
                seed_data.append(str(round(time.time() - ss_start_time + ss_acc_time, 4)) + '\t' + str(mep_k_prod) + '\t' + str(mep_i_node) + '\t' +
                                 str(now_budget) + '\t' + str([len(seed_set[k]) for k in range(num_product)]) + '\n')

                if self.diff_seed_option:
                    random_node_list = [celf_item for celf_item in random_node_list if celf_item[1] != mep_i_node]

            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
            print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
            seed_set_sequence[now_bi_index] = seed_set
            ss_time_sequence[now_bi_index] = ss_time
            seed_data_sequence[now_bi_index] = seed_data

            for wd in wd_seq:
                seed_data_path = 'seed_data/' + self.model_name + '_' + wd
                if not os.path.isdir(seed_data_path):
                    os.mkdir(seed_data_path)
                seed_data_file = open(seed_data_path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name +
                                      '_' + self.seed_cost_option + '_bi' + str(self.budget_iteration[now_bi_index]) + '.txt', 'w')
                for sd in seed_data:
                    seed_data_file.write(sd)
                seed_data_file.close()

        while -1 in seed_data_sequence:
            no_data_index = seed_data_sequence.index(-1)
            seed_data_sequence[no_data_index] = seed_data_sequence[no_data_index - 1]
            ss_time_sequence[no_data_index] = ss_time_sequence[no_data_index - 1]

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.seed_cost_option, self.diff_seed_option, self.cascade_model)
        for bi in self.budget_iteration:
            now_bi_index = self.budget_iteration.index(bi)
            if self.wallet_distribution_type:
                eva_model.evaluate(bi, self.wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])
            else:
                for wallet_distribution_type in self.wd_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, seed_set_sequence[now_bi_index], ss_time_sequence[now_bi_index])