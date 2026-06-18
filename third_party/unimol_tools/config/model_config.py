MODEL_CONFIG = {
    "weight":{
        "protein": "pocket_pre_220816.pt",
        "small_molecule_no_h": {
            "none": None,
            "pretrain_self_supervised": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/mol_pre_no_h_220816.pt",
        },
        "small_molecule_all_h":  {
            "none": None,
            "pretrain_self_supervised": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/mol_pre_all_h_220816.pt",
            "battery": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/batterypbc_singleatom_cutoff6_model_3.pth",
            "qm9nmr_all": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/qm9nmr_all_epoch40.pth",
            "nmrshiftdb2_all": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/nmrshiftdb2_all_epoch50.pth",
            # "nmrshiftdb2_all": "/vepfs/fs_users/yongqi/UniNMR/results/baseline/weight_small_molecule_pretrain_self_supervised_ALL_sampling_ratio_None_batchsize_8_gradaccum_1_epochs_50_optimizer_adam_lr_0.0001_loss_l1_metric_mae_onlyatom_earlystopping_10/model_0.pth",
            "ACD_ALL": "/vepfs/fs_users/yongqi/UniNMR/results/ACD_ALL/weight_small_molecule_pretrain_self_supervised_ALL_sampling_ratio_None_batchsize_8_gradaccum_1_epochs_20_optimizer_adam_lr_0.0001_loss_l1_metric_mae_onlyatom_earlystopping_10/model_0.pth",
            "ACD_C": "/vepfs/fs_users/yongqi/UniNMR/results/ACD_C/weight_small_molecule_pretrain_self_supervised_C_sampling_ratio_None_batchsize_8_gradaccum_1_epochs_20_optimizer_adam_lr_0.0001_loss_l1_metric_mae_onlyatom_earlystopping_10/model_0.pth",
            "ACD_H": "/vepfs/fs_users/yongqi/UniNMR/results/ACD_H/weight_small_molecule_pretrain_self_supervised_H_sampling_ratio_None_batchsize_8_gradaccum_1_epochs_20_optimizer_adam_lr_0.0001_loss_l1_metric_mae_onlyatom_earlystopping_10/model_0.pth",
        },
        "inorganic_crystal": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/mp_all_h_230313.pt",
        "organic_crystal": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/pretrain_csd_limit_from_mol_0.2atom6_global.pt",
        "mof": "/vepfs/fs_users/yongqi/ckps/UniNMR/saved/unimof_pretrain_best.pt",
    },
    "dict":{
        "protein": "poc.dict.txt",
        "small_molecule_no_h": "mol.dict.txt",
        "small_molecule_all_h": "mol.dict.txt",
        "inorganic_crystal": "mp.dict.txt",
        "organic_crystal": "oc_limit_dict.txt",
        "mof": "mof.dict.txt",
    },
}