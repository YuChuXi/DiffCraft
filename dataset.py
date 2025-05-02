import os
import numpy
import torch
import logging
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from litemapy import Schematic, Region, BlockState

logger = logging.getLogger(__name__)

LITEMATIC = ".litematic"
SAVED = ".npz" #".pth" # pth保存的非常大

class BlockStateConverter:
    def __init__(self, block_table_path: str, state_table_path: str):
        self.block_table = {}
        self.state_table = {}

        with open(block_table_path, "r") as f:
            for line in f.readlines():
                idx, block = line.strip().split("\t")
                self.block_table[block] = int(idx)

        with open(state_table_path, "r") as f:
            lines = f.readlines()
            self.max_state_num = int(lines.pop(0).split(":")[1])
            for line in lines:
                idx, state = line.strip().split("\t")
                self.state_table[state] = int(idx)
                
        logger.info(
            f"Loaded {len(self.block_table)} blocks and {len(self.state_table)} states"
        )

    # 0 is reserved for "none"
    def get_block_idx(self, block: BlockState) -> int:
        if block.id not in self.block_table:
            logger.warning(f"Block {block.id} not found in block table")
            return 0
        return self.block_table[block.id]

    def get_block_state_idx(self, block: BlockState) -> int:
        if block.id not in self.block_table:
            logger.warning(f"Block {block.id} not found in block table")
            return set()
        states = []
        for s in self.get_block_state(block):
            if s not in self.state_table:
                logger.warning(f"State {s} not found in state table")
                continue
            states.append(self.state_table[s])
        return set(states)

    @staticmethod
    def get_block_state(block: BlockState) -> set:
        rawstate = block.to_block_state_identifier()
        if "[" not in rawstate:
            return set()
        return set(rawstate.split("[")[1].split("]")[0].split(","))

class LitematicaDataset(Dataset):
    def __init__(self, cache_pth: bool = True, all_in_mem: bool = True):
        super().__init__()

        self.block_state_converter = BlockStateConverter(
            "assets/blocks.txt", "assets/states.txt"
        )
        self.sample_name_list = list(
            dict.fromkeys(
                [
                    i[: -len(LITEMATIC)]
                    for i in os.listdir("dataraw")
                    if i.endswith(LITEMATIC)
                ]
                + [i[: -len(SAVED)] for i in os.listdir("dataset") if i.endswith(SAVED)]
            )
        )
        self.sample_map_dict = {}

        if (not cache_pth) and (not all_in_mem):
            return  # early exit!

        for s in tqdm(self.sample_name_list, desc="Loading litematics"):
            try:
                assert os.path.exists(f"dataset/{s}{SAVED}")
                if SAVED == ".pth":
                    map = torch.load(f"dataset/{s}{SAVED}")
                elif SAVED == ".npz":
                    map = numpy.load(f"dataset/{s}{SAVED}")["map"]
            except Exception as e:
                map = self.load_litematic(f"dataraw/{s}{LITEMATIC}")

            if all_in_mem:
                self.sample_map_dict[s] = map
            if cache_pth:
                if SAVED == ".pth":
                    torch.save(map, f"dataset/{s}{SAVED}")
                elif SAVED == ".npz":
                    numpy.savez_compressed(f"dataset/{s}{SAVED}", map=map)

    def load_litematic(self, file):
        schematic = Schematic.load(file)
        MAX_STATE_NUM = self.block_state_converter.max_state_num
        
        map = torch.zeros(
            (schematic.width, schematic.height, schematic.length, MAX_STATE_NUM+1), dtype=torch.uint16
        )  # (x, y, z, (id, state))
        for name, reg in schematic.regions.items():
            for pos in reg.allblockpos():
                block = reg.getblock(*pos)
                pos = (pos[0] + reg.x, pos[1] + reg.y, pos[2] + reg.z)
                map[*pos, 0] = self.block_state_converter.get_block_idx(block)
                state = self.block_state_converter.get_block_state_idx(block)
                for i, s in enumerate(state):
                    assert i < MAX_STATE_NUM, f"Too many states for block {block.id} at {pos}, {state}"
                    map[pos[0], pos[1], pos[2], i+1] = s
        return map

    def __len__(self):
        return len(self.sample_name_list)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.sample_name_list[idx]
        elif isinstance(idx, str):
            name = idx
        
        if name in self.sample_map_dict:
            return self.sample_map_dict[name]
        
        if not os.path.exists(f"dataset/{name}{SAVED}"):
            raise FileNotFoundError(f"File {name}{SAVED} not found")
        
        if SAVED == ".pth":
            return torch.load(f"dataset/{name}{SAVED}")
        elif SAVED == ".npz":
            return numpy.load(f"dataset/{name}{SAVED}")["map"]
        