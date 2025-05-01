from dataset import LitematicaDataset

dataset = LitematicaDataset()

#print(dataset.sample_name_list)
#print(dataset.sample_map_list)

for i in dataset.sample_map_list:
    #print(i)
    print(dataset.sample_map_list[i].shape)
    print(dataset.sample_map_list[i].dtype)
    print(dataset.sample_map_list[i].device)