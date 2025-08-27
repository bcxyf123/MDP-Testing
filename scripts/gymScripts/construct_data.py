import numpy as np
import csv

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import csv

# 创建横坐标
x = np.arange(0.1, 10.1, 0.1)

y = np.zeros_like(x)
noise = np.random.normal(0, 0.15, size=y.shape)
y = np.clip(y + noise, 0, 1)

# 绘制曲线
plt.plot(x, y)
plt.show()

# 创建csv文件
with open('tables/MPE_reward/pedm.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入列标题
    writer.writerow(["powers"])
    # 写入纵坐标
    for data in y:
        writer.writerow([data])