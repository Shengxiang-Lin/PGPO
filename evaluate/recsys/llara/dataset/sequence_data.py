import json
import os.path as op
import random

import pandas as pd
import torch.utils.data as data


class SequenceData(data.Dataset):
    def __init__(
        self,
        data_dir="./data/ml-1m/v1",
        stage=None,
        cans_num=10,
        sep=", ",
        no_augment=True,
    ):
        self.__dict__.update(locals())
        self.aug = (stage == "train") and not no_augment
        self.check_files()

    def __len__(self):
        return len(self.session_data["seq"])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp["seq_unpad"], temp["next"])
        sample = {
            "seq": temp["seq"],
            "seq_name": [f"item_{item_id}" for item_id in temp["seq_unpad"]],
            "len_seq": temp["len_seq"],
            "seq_str": self.sep.join([f"item_{item_id}" for item_id in temp["seq_unpad"]]),
            "cans": candidates,
            "cans_name": [f"item_{item_id}" for item_id in candidates],
            "cans_str": self.sep.join([f"item_{item_id}" for item_id in candidates]),
            "len_cans": self.cans_num,
            "item_id": temp["next"],
            "item_name": f"item_{temp['next']}",
            "correct_answer": f"item_{temp['next']}",
        }
        return sample

    def negative_sampling(self, seq_unpad, next_item):
        canset = [i for i in range(self.num_items) if i not in seq_unpad and i != next_item]
        if len(canset) < self.cans_num - 1:
            candidates = random.choices(canset, k=self.cans_num - 1) + [next_item]
        else:
            candidates = random.sample(canset, self.cans_num - 1) + [next_item]
        random.shuffle(candidates)
        return candidates

    def check_files(self):
        meta_path = op.join(self.data_dir, "meta.json")
        if not op.exists(meta_path):
            raise FileNotFoundError(f"meta file not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.padding_item_id = int(meta["padding_item_id"])
        self.num_items = int(meta["num_items"])
        if self.stage == "train":
            filename = "train_data.df"
        elif self.stage == "val":
            filename = "Val_data.df"
        elif self.stage == "test":
            filename = "Test_data.df"
        else:
            raise ValueError("stage must be train/val/test")
        data_path = op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path)

    def session_data4frame(self, datapath):
        session_data = pd.read_pickle(datapath)
        session_data = session_data[session_data["len_seq"] >= 3]

        def get_id_from_tuple(x):
            return int(x[0]) if isinstance(x, (tuple, list)) else int(x)

        def get_ids_from_list(x):
            return [int(i[0]) if isinstance(i, (tuple, list)) else int(i) for i in x]

        session_data["next"] = session_data["next"].apply(get_id_from_tuple)
        session_data["seq"] = session_data["seq"].apply(get_ids_from_list)

        def remove_padding(x):
            return [int(i) for i in x if int(i) != self.padding_item_id]

        session_data["seq_unpad"] = session_data["seq"].apply(remove_padding)
        return session_data
