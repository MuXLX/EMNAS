import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compare(p1, p2):
    # return 0同层 1 p1支配p2
    # 每个维度越小越优秀
    # 计D次
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True

    for i in range(2):
        if p1[i + 1] > p2[i + 1]:
            p1_dominate_p2 = False
        if p1[i + 1] < p2[i + 1]:
            p2_dominate_p1 = False

    # 代表两者相等
    if p1_dominate_p2 == p2_dominate_p1:
        return 0
    else:
        return 1 if p1_dominate_p2 else -1


def non_dominated_sort(P):
    # 成员编号为 0 ~ P_size-1
    P_size = len(P)
    # 被支配数
    n = np.full(shape=P_size, fill_value=0)
    # 支配的成员
    S = []
    # 每层包含的成员编号们
    f = []  # 0 开始
    dominated = []
    # 所处等级
    rank = np.full(shape=P_size, fill_value=-1)

    f_0 = []
    non_dominated = []
    # 对每一个 个体分别进行非支配的查看
    for p in range(P_size):
        # 第p个个体被支配的次数，用于层数划分
        n_p = 0
        # 第p个个体支配的个体集合
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            # p支配q
            if cmp == 1:
                S_p.append(q)
            # p被q支配
            elif cmp == -1:  # 被支配
                n_p += 1
        # 代表第p个个体支配的成员
        S.append(S_p)
        # 第p个个体被支配的次数
        n[p] = n_p
        # 代表当前个体为非支配个体
        if n_p == 0:
            # P[p][4] = 0
            rank[p] = 0
            # 加入非支配集合
            f_0.append(p)
            non_dominated.append(P[p])

    # 重置非支配集合

    f.append(f_0)  # 这时候f[0]必存在
    dominated.append(non_dominated)

    i = 0
    while len(f[i]) != 0:  # 可能还有i+1层
        Q = []
        D = []
        # 获取当前层个体
        for p in f[i]:  # i层中每个个体
            # 去除当前个体时，为该个体所支配的个体的被支配次数-1
            for q in S[p]:  # 被p支配的个体
                n[q] -= 1
                if n[q] == 0:
                    # 该个体属于i+1层
                    rank[q] = i + 1
                    # P[q][4] = i + 1
                    Q.append(q)
                    D.append(P[q])
        i += 1
        f.append(Q)
        dominated.append(D)

    return dominated


def crowding_distance_assignment(Di,t):
    s = sorted(Di, key=lambda stu : stu[3])
    l = len(s)
    s1_max = s[0][3]
    s1_min = s[l-1][3]
    s2_max = -1
    s2_min = 1000
    for i in s:
        if i[4] > s2_max:
            s2_max = i[4]
        if i[4] < s2_min:
            s2_min = i[4]
    S1 = s1_max-s1_min
    S2 = s2_max - s2_min
    if S1 == 0:
        S1 = 1
    if S2 == 0:
        S2 = 1
    while l > t:
        distant = [] + [0]*l
        distant[0] = 1000000
        distant[l-1] = 1000000

        for i in range(1,l-1):
            distant[i] = distant[i] + abs(s[i - 1][4] - s[i + 1][4]) / S2 + abs(s[i - 1][3] - s[i + 1][3]) / S1
            # distant[i] = distant[i] + abs(s[i - 1][1] - s[i + 1][1]) / S1

        min_index = distant.index(min(distant))
        del s[min_index]
        l = len(s)
    return s


W = np.loadtxt('/home/ljh/Code/NAS/Evolution/Darts/Run_M3/dim4.csv')
data = pd.read_excel('/home/ljh/Code/NAS/Evolution/Darts/Run_M4/Result/All1.xlsx')
data = data.iloc[:,[2,4,5]]

s = data.to_numpy().tolist()
# s = non_dominated_sort(s)
# print(s[0])

# self.k = len(self.W)

A = [[],[],[],[],[],[],[],[],[]]
# 对所有个体进行分区域
for p in s:
    # 提取三个目标值
    l = p
    ma = -1
    t = -1
    # 计算内积得知夹角，从而将个体分配到i类中
    for i, w in enumerate(W):
        dot = np.dot(l[1:3], w)
        if dot > ma:
            ma = dot
            t = i
    A[t].append(l)

# s = data.to_numpy().tolist()
# print(s)
S = non_dominated_sort(s)
print(S[0])
for i in range(len(W)):
    t = A[i]
    # print(t)
    k = non_dominated_sort(t)
    print(k[0])


