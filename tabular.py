# %%
import numpy as np
import torch

# %%
data_np = np.loadtxt("./traffic.csv", dtype=np.int64, delimiter=",", skiprows=1, converters={
    0: lambda x: 
    np.int64(x.split()[-1][0:2])}
    )
data_np

# %%
data_t = torch.from_numpy(data_np)
data_no_time = data_t[:, 1:]
data_no_time

# %%
data_reshaped = data_no_time.view(-1, 24, data_no_time.shape[1]).transpose(1,2)
data_reshaped.shape

# %%
firstDay = data_reshaped[0]
firstDayJunction = firstDay[0]
first_day_junction_onehot = torch.zeros(firstDay.shape[1], 4, dtype=torch.uint8)
first_day_junction_onehot.scatter_(1, firstDayJunction.unsqueeze(-1) - 1, 1)

torch.cat([firstDay, first_day_junction_onehot.transpose(0,1)], dim=0)


# %%
all_junctions = data_reshaped[:,0]
junction_onehot = torch.zeros(data_reshaped.shape[0], data_reshaped.shape[2], 4, dtype=torch.uint8)
junction_onehot.scatter_(2, all_junctions.unsqueeze(-1) - 1, 1)

torch.cat([data_reshaped, junction_onehot.transpose(1,2)], 1)

# %%
data_reshaped = data_reshaped.to(torch.float32)
data_reshaped = (data_reshaped - data_reshaped.mean()) / data_reshaped.std()
data_reshaped

# %%



