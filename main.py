from Model import *

if __name__ == '__main__':
    sc_option_seq = [1, 2, 3]
    ds_option_seq = [2]
    dataset_seq = [1]
    prod_seq = [1]
    cm_seq = [1, 2]
    wd_seq = [1, 2]

    for sc_option in sc_option_seq:
        seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
        for ds_option in ds_option_seq:
            diff_seed_option = False if ds_option == 1 else True
            for data_setting in dataset_seq:
                dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                               'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
                for cm in cm_seq:
                    cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
                    for prod_setting in prod_seq:
                        product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                        Model('mmioa', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_mioa(r_flag=False)
                        Model('mmioar', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_mioa(r_flag=True)
                        Model('mdag1', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag1(r_flag=False)
                        Model('mdag1r', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag1(r_flag=True)
                        Model('mdag2', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag2(r_flag=False)
                        Model('mdag2r', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag2(r_flag=True)
                        Model('mng', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_ng(r_flag=False)
                        Model('mngr', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_ng(r_flag=True)
                        Model('mhd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_hd()
                        Model('mr', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_r()
                        # Model('mpmis', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_pmis()

                        for wd in wd_seq:
                            wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                            Model('mmioaepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=False, epw_flag=True)
                            Model('mmioarepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=True, epw_flag=True)
                            Model('mdag1epw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=False, epw_flag=True)
                            Model('mdag1repw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=True, epw_flag=True)
                            Model('mdag2epw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=False, epw_flag=True)
                            Model('mdag2repw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=True, epw_flag=True)
                            Model('mmioapw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=False)
                            Model('mmioarpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=True)
                            Model('mdag1pw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=False)
                            Model('mdag1rpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=True)
                            Model('mdag2pw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=False)
                            Model('mdag2rpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=True)
                            Model('mngpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_ng(r_flag=False)
                            Model('mngrpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_ng(r_flag=True)