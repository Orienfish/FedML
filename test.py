
from fedml_api.data_preprocessing.load_data import load_partition_data_HPWREN


(
    train_data_num,
    test_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    output_dim,
) = load_partition_data_HPWREN(50,"data/HPWREN")













