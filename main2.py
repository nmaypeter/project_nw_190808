from Model import *

if __name__ == '__main__':
    sc_option_seq = [1, 2, 3]
    ds_option_seq = [1, 2]
    dataset_seq = [1, 2, 3, 4]
    prod_seq = [1, 2]
    cm_seq = [1, 2]
    wd_seq = [1, 2]

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                       'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
        for sc_option in sc_option_seq:
            seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
            for ds_option in ds_option_seq:
                diff_seed_option = False if ds_option == 1 else True
                for cm in cm_seq:
                    cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
                    for prod_setting in prod_seq:
                        product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                        Model('mmioad', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_mioa(r_flag=False, d_flag=True)
                        Model('mmioard', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_mioa(r_flag=True, d_flag=True)
                        Model('mdag1d', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag1(r_flag=False, d_flag=True)
                        Model('mdag1rd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag1(r_flag=True, d_flag=True)
                        Model('mdag2d', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag2(r_flag=False, d_flag=True)
                        Model('mdag2rd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_dag2(r_flag=True, d_flag=True)
                        Model('mngd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_ng(r_flag=False, d_flag=True)
                        Model('mngrd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_ng(r_flag=True, d_flag=True)
                        # Model('mhd', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_hd()
                        # Model('mr', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option).model_r()

                        for wd in wd_seq:
                            wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                            Model('mmioadepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=False, d_flag=True, epw_flag=True)
                            Model('mmioardepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=True, d_flag=True, epw_flag=True)
                            Model('mdag1depw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=False, d_flag=True, epw_flag=True)
                            Model('mdag1rdepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=True, d_flag=True, epw_flag=True)
                            Model('mdag2depw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=False, d_flag=True, epw_flag=True)
                            Model('mdag2rdepw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=True, d_flag=True, epw_flag=True)
                            Model('mmioadpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=False, d_flag=True)
                            Model('mmioardpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_mioa(r_flag=True, d_flag=True)
                            Model('mdag1dpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=False, d_flag=True)
                            Model('mdag1rdpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag1(r_flag=True, d_flag=True)
                            Model('mdag2dpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=False, d_flag=True)
                            Model('mdag2rdpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_dag2(r_flag=True, d_flag=True)
                            Model('mngdpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_ng(r_flag=False, d_flag=True)
                            Model('mngrdpw', dataset_name, product_name, cascade_model, seed_cost_option, diff_seed_option, wallet_distribution_type).model_ng(r_flag=True, d_flag=True)