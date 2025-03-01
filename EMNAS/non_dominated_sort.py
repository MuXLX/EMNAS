import numpy as np

# 判断p1是否支配p2
def compare(p1, p2):
    # return 0同层 1 p1支配p2
    # 每个维度越小越优秀
    # 计D次
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True

    for i in range(2):
        if p1[i+1] > p2[i+1]:
            p1_dominate_p2 = False
        if p1[i+1] < p2[i+1]:
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
                    Q.append(q)
                    D.append(P[q])
        i += 1
        f.append(Q)
        dominated.append(D)

    return dominated


def crowding_distance_assignment(Di,t):
    s = sorted(Di, key=lambda stu : stu[1])
    l = len(s)
    # print(s)
    s1_max = s[0][1]
    s1_min = s[l-1][1]
    s2_max = -1
    s2_min = 1000
    for i in s:
        if i[2] > s2_max:
            s2_max = i[2]
        if i[2] < s2_min:
            s2_min = i[2]
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
            distant[i] = distant[i] + abs(s[i - 1][2] - s[i + 1][2]) / S2 + abs(s[i - 1][1] - s[i + 1][1]) / S1
            # distant[i] = distant[i] + abs(s[i - 1][1] - s[i + 1][1]) / S1

        min_index = distant.index(min(distant))
        del s[min_index]
        l = len(s)
    return s