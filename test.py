'''
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare


(
    client_num,
    train_data_num,
    test_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    output_dim,
) = load_partition_data_shakespeare(10)



print("=============================client_num")
print(client_num)
print("=========================train_data_num")
print(train_data_num)
print("==========================test_data_num")
print(test_data_num)
print("=============================output_dim")
print(output_dim)
print("loaded")
'''











