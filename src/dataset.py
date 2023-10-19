from torch.utils.data import Dataset


class BLI_Dataset(Dataset):
    def __init__(self, data):
        self.query_names = data
    def __getitem__(self, query_idx):
        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        return query_name1, query_name2, query_idx
    def __len__(self):
        return len(self.query_names)                                         
