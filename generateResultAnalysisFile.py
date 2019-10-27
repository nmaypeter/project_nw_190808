sc_option_seq = [1, 2, 3]
cm_seq = [1, 2]
dataset_seq = [1, 2, 3, 4]
prod_seq = [1, 2]
wallet_distribution_seq = [1, 2]
model_seq = ['mmioaepw', 'mmioarepw', 'mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
             'mmioa', 'mmioapw', 'mmioar', 'mmioarpw',
             'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
             'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
             'mng', 'mngpw', 'mngr', 'mngrpw', 'mhd', 'mr']
num_product = 3

for sc_option in sc_option_seq:
    seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        profit_list, cost_list, time_list = [], [], []
        # number_seed_list = []
        for bi in range(10, 5, -1):
            for data_setting in dataset_seq:
                dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                               'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
                for prod_setting in prod_seq:
                    product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                    for wallet_distribution in wallet_distribution_seq:
                        wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)

                        profit, cost, time = '', '', ''
                        # number_seed = ['' for _ in range(num_product)]
                        r = dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                            wallet_distribution_type + '\t' + product_name + '\t' + str(bi)
                        print(r)
                        for model_name in model_seq:
                            try:
                                result_name = 'result/' + \
                                              model_name + '_' + wallet_distribution_type + '/' + \
                                              dataset_name + '_' + cascade_model + '_' + product_name + '_' + seed_cost_option + \
                                              '_bi' + str(bi) + '.txt'

                                with open(result_name) as f:
                                    p = 0
                                    for lnum, line in enumerate(f):
                                        if lnum < 3 or 5 < lnum < 9:
                                            continue
                                        elif lnum == 3:
                                            (l) = line.split()
                                            p = float(l[-1])
                                        elif lnum == 4:
                                            (l) = line.split()
                                            c = float(l[-1])
                                            profit += str(round(p - c, 4)) + '\t'
                                            cost += l[-1] + '\t'
                                        elif lnum == 5:
                                            (l) = line.split()
                                            time += l[-1] + '\t'
                                        # elif lnum == 9:
                                        #     (l) = line.split()
                                        #     for nl in range(2, len(l)):
                                        #         number_seed[nl-2] += l[nl] + '\t'
                                        else:
                                            break
                            except FileNotFoundError:
                                profit += '\t'
                                cost += '\t'
                                time += '\t'
                                # for nl in range(num_product):
                                #     number_seed[nl] += '\t'
                        profit_list.append(profit)
                        cost_list.append(cost)
                        time_list.append(time)
                        # for nl in range(num_product):
                        #     number_seed_list.append(number_seed[nl])

        path = 'result/result_analysis_' + seed_cost_option + '_' + cascade_model
        fw = open(path + '_profit.txt', 'w')
        for lnum, line in enumerate(profit_list):
            if lnum % (len(prod_seq) * len(wallet_distribution_seq)) == 0 and lnum != 0:
                fw.write('\n')
            fw.write(str(line) + '\n')
        fw.close()
        fw = open(path + '_cost.txt', 'w')
        for lnum, line in enumerate(cost_list):
            if lnum % (len(prod_seq) * len(wallet_distribution_seq)) == 0 and lnum != 0:
                fw.write('\n')
            fw.write(str(line) + '\n')
        fw.close()
        fw = open(path + '_time.txt', 'w')
        for lnum, line in enumerate(time_list):
            if lnum % (len(prod_seq) * len(wallet_distribution_seq)) == 0 and lnum != 0:
                fw.write('\n')
            fw.write(str(line) + '\n')
        # fw = open(path + '_num_seed.txt', 'w')
        # for lnum, line in enumerate(number_seed_list):
        #     if lnum % (len(prod_seq) * len(wallet_distribution_seq) * num_product) == 0 and lnum != 0:
        #         fw.write('\n')
        #     fw.write(str(line) + '\n')