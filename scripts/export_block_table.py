from natsort import natsorted
from litemapy import Schematic, Region, BlockState
from dataset import BlockStateConverter

schematic = Schematic.load("assets/Debug.litematic")
reg = schematic.regions[list(schematic.regions.keys())[0]]

blocks = set()
states = set()
max_state_num = 0
for b in reg.palette:
    blocks.add(b.id)
    state = BlockStateConverter.get_block_state(b)
    states.update(state)
    max_state_num = max(max_state_num, len(state))

print(f"Max state num: {max_state_num}")

blocks.remove("minecraft:air") # 确保air在第一个
blocks = natsorted(list(blocks))
blocks.insert(0, "minecraft:air")

states = natsorted(list(states)) # 保留一个占位符
states.insert(0, "none")

with open("assets/blocks.txt", "w") as f:
    for idx, b in enumerate(blocks):
        f.write(f"{idx}\t{b}\n")
        
with open("assets/states.txt", "w") as f:
    f.write(f"# max_state_num: {max_state_num} \n")
    for idx, s in enumerate(states):
        f.write(f"{idx}\t{s}\n")
