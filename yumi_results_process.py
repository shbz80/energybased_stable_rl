from energybased_stable_rl.seed_aggregation import aggregate_results as yumi_aggregate_results
success_dit = 0.004

############### yumi ANN-PPO ##########################
exps = ['elitebook_ral_revised/yumipeg_ppo_garage',
        'elitebook_ral_revised/yumipeg_ppo_garage_1',
        'elitebook_ral_revised/yumipeg_ppo_garage_2',
        'elitebook_ral_revised/yumipeg_ppo_garage_3',
        'elitebook_ral_revised/yumipeg_ppo_garage_4']
file_name = 'yumi_ann'
# yumi_aggregate_results(exps, file_name, success_dit)
print('ANN-PPO OK')

############### yumi NF-CEM-k1 ##########################
exps = ['cem_nf_yumi_1',
        'cem_nf_yumi_2',
        'cem_nf_yumi_3',
        'cem_nf_yumi_4',
        'cem_nf_yumi_5']
file_name = 'yumi_nf_k1'
yumi_aggregate_results(exps, file_name, success_dit, flag=True)
print('NF-CEM-K1 OK')

############### yumi NF-CEM-k4 ##########################
exps = ['elitebook_ral_revised/cem_nf_yumi_1',
        'elitebook_ral_revised/cem_nf_yumi_2',
        'elitebook_ral_revised/cem_nf_yumi_3',
        'elitebook_ral_revised/cem_nf_yumi_4',
        'elitebook_ral_revised/cem_nf_yumi_5']
file_name = 'yumi_nf_k4'
# yumi_aggregate_results(exps, file_name, success_dit)
print('NF-CEM-K2 OK')

############### yumi ES-CEM ##########################
exps = ['elitebook_ral_revised/cem_energybased_yumi',
        'elitebook_ral_revised/cem_energybased_yumi_1',
        'elitebook_ral_revised/cem_energybased_yumi_5',
        'elitebook_ral_revised/cem_energybased_yumi_6',
        'elitebook_ral_revised/cem_energybased_yumi_8']
file_name = 'yumi_es'
# yumi_aggregate_results(exps, file_name, success_dit)
print('ES-CEM OK')
