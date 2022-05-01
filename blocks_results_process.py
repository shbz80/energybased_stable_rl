from seed_aggregation import aggregate_results as blocks_aggregate_results
success_dit = 0.025
############### block 2D ANN-PPO ##########################
exps = ['block2D_ppo_torch_garage',
        'block2D_ppo_torch_garage_1',
        'elitebook_ral_revised/block2D_ppo_torch_garage_5',
        'elitebook_ral_revised/block2D_ppo_torch_garage_6',
        'elitebook_ral_revised/block2D_ppo_torch_garage_7']
file_name = 'blocks_ann'
blocks_aggregate_results(exps, file_name, success_dit)

############### block 2D NF-CEM-k1 ##########################
exps = ['cem_nf_block2d_3',
        'cem_nf_block2d_4',
        'cem_nf_block2d_5',
        'cem_nf_block2d_6',
        'cem_nf_block2d_7']
file_name = 'blocks_nf_k1'
blocks_aggregate_results(exps, file_name, success_dit)

############### block 2D NF-CEM-k2 ##########################
exps = ['elitebook_ral_revised/cem_nf_block2d',
        'elitebook_ral_revised/cem_nf_block2d_1',
        'elitebook_ral_revised/cem_nf_block2d_2',
        'elitebook_ral_revised/cem_nf_block2d_3',
        'elitebook_ral_revised/cem_nf_block2d_4']
file_name = 'blocks_nf_k2'
blocks_aggregate_results(exps, file_name, success_dit)

############### block 2D ES-CEM ##########################
exps = ['cem_energybased_block2d_12',
        'cem_energybased_block2d_13',
        'cem_energybased_block2d_14',
        'cem_energybased_block2d_15',
        'cem_energybased_block2d_16']
file_name = 'blocks_es'
blocks_aggregate_results(exps, file_name, success_dit)

############### block 2D ES-CEM-ICNN ##########################
exps = ['cem_energybased_block2d_17',
        'cem_energybased_block2d_21',
        'cem_energybased_block2d_22',
        'cem_energybased_block2d_23',
        'cem_energybased_block2d_24']
file_name = 'blocks_es_icnn'
blocks_aggregate_results(exps, file_name, success_dit)

############### block 2D ES-CEM-QUAD ##########################
exps = ['cem_energybased_block2d_18',
        'cem_energybased_block2d_19',
        'cem_energybased_block2d_20']
file_name = 'blocks_es_quad'
blocks_aggregate_results(exps, file_name, success_dit)
