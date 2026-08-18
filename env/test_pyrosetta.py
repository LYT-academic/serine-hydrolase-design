import pyrosetta
pyrosetta.init()

# 创建一个简单的多肽结构
pose = pyrosetta.pose_from_sequence("TEST")

# 加载打分函数
scorefxn = pyrosetta.get_fa_scorefxn()

# 计算打分
print("Total score:", scorefxn(pose))
