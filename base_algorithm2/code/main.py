from copy import deepcopy
import numpy as np
import random
import time

def is_diff_lane(id1, id2, lane_list):
    if id1 in lane_list and id2 not in lane_list:
        return True
    elif id1 not in lane_list and id2 in lane_list:
        return True
    else:
        return False




def processA_hv(A1, A2, hvlist):  # 算法主要部分
    A1_time = {i: A1[i][0] / (A1[i][1] + 0.01) for i in A1.keys()}
    A2_time = {i: A2[i][0] / (A2[i][1] + 0.01) for i in A2.keys()}
    A_time = {**A1_time, **A2_time}
    A1_id = sorted(A1_time, key=lambda k: A1_time[k])
    A2_id = sorted(A2_time, key=lambda k: A2_time[k])
    A_id = sorted(A_time, key=lambda k: A_time[k])
    hv1_id = [i for i in A1_id if i in hvlist]
    hv2_id = [i for i in A2_id if i in hvlist]
    standard_scheme_A = deepcopy(A_id)  # 先到先得顺序

    standard_timetable_A = generate_timetable(standard_scheme_A, A_time, hvlist)  # 记录在FIFS顺序下的到达时间
    all_scheme_A = generate_permutations(A1_id, A2_id)  # 所有可能顺序
    # 剔除不符合hv要求的顺序
    wrong_scheme = []
    for s in all_scheme_A:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于3.36s，则认为方案不行
            p = s.index(vid)
            if p != 0 and s[p - 1] in A2_id and A_time[s[p - 1]] - A_time[vid] < 3.36:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于3.36s，则认为不行
            p = s.index(vid)
            if p != 0 and p != len(s) - 1 and s[p - 1] in A1_id and s[p + 1] in A1_id and A_time[s[p + 1]] - A_time[
                s[p - 1]] < 3.36:
                wrong_scheme.append(s)
                break
    for s in wrong_scheme:
        all_scheme_A.remove(s)
    U_A_scheme, T_A_scheme = {}, {}
    for scheme_i in all_scheme_A:
        utility, timetable = pruner(scheme_i, A_time, standard_timetable_A, A1_id)
        if utility is not None:
            U_A_scheme[tuple(scheme_i)] = utility
            T_A_scheme[tuple(scheme_i)] = timetable
    if len(T_A_scheme) == 0:
        return [], U_A_scheme, T_A_scheme, standard_timetable_A
    else:
        scheme_A = list(U_A_scheme.keys())
        # P_A_scheme = {sc: {vehid: U_A_scheme[sc][vehid] / sum(i[vehid] for i in U_A_scheme.values()) for vehid in A_id} for sc in scheme_A}
        return scheme_A, U_A_scheme, T_A_scheme, standard_timetable_A

def generate_timetable(scheme: list, a_time: dict, hvlist=[]) -> dict:  # 计算通行时间表
    """
    :param scheme: 列表，通行顺序方案
    :param a_time: 字典，默认情况下车辆到达时间
    :return: 字典，根据通行顺序列表修正后的达到时间
    """
    timetable = {}
    for i, vehid in enumerate(scheme):
        if i == 0:
            if scheme[i] in hvlist:
                timetable[vehid] = a_time[vehid]
            else:
                timetable[vehid] = max(a_time[vehid] * 0.9, min(a_time[vehid], 1))
        else:
            if scheme[i - 1] in hvlist:
                timetable[vehid] = max(timetable[scheme[i - 1]] + 1.7, a_time[vehid])
            else:
                timetable[vehid] = max(timetable[scheme[i - 1]] + 1.4, a_time[vehid] * 0.9)
    return timetable

def generate_permutations(list1, list2):   # 用于求解全排列    
    def backtrack(idx1, idx2, curr_permutation):
        if idx1 == len(list1) and idx2 == len(list2):
            permutations.append(curr_permutation[:])
            return
        if idx1 < len(list1):
            curr_permutation.append(list1[idx1])
            backtrack(idx1 + 1, idx2, curr_permutation)
            curr_permutation.pop()
        if idx2 < len(list2):
            curr_permutation.append(list2[idx2])
            backtrack(idx1, idx2 + 1, curr_permutation)
            curr_permutation.pop()

    permutations = []
    backtrack(0, 0, [])
    return permutations

def pruner(scheme: list, a_time, std_timetable, list1, t_res=4.5, hvlist=[]):  # 剪枝操作
    """
    :param scheme: 列表，由id构成的通行方案
    :param a_time: 字典，默认情况下的到达时间
    :param std_timetable: 字典，FIFS下的到达时间
    :param list1: 其中一个方向的所有车辆ID
    :param t_res: 允许变化的最大时间延误
    :return: 字典，返回合理方案对应的效用；方案不合理则返回None
    """
    timetable = generate_timetable(scheme, a_time, hvlist)  # 计算新方案下的到达时间
    # 计算基础收益值
    utility = {}
    for order, vid in enumerate(scheme):
        t1 = timetable[vid]
        t2 = std_timetable[vid]
        if t1 - t2 >= t_res:
            return None, timetable
        elif 3.5 <= t1 - t2:
            if order >= 2 and is_diff_lane(vid, scheme[order - 1], list1):
                utility[vid] = np.exp((t2 - t1) / 2)
            else:
                return None, timetable
        else:
            utility[vid] = np.exp((t2 - t1) / 2)
    # 计算修正收益值
    ans_utility = {}
    for i in range(len(scheme)):
        if i > 0 and not is_diff_lane(scheme[i - 1], scheme[i], list1):
            # 有前车
            d1 = max(min((a_time[scheme[i]] - a_time[scheme[i - 1]]) / 3, 1), 0.5)
            ans_utility[scheme[i]] = d1 / 2 * utility[scheme[i]] + (1 - d1) / 2 * utility[scheme[i - 1]]
        else:
            ans_utility[scheme[i]] = utility[scheme[i]] / 2

        if i < len(scheme) - 1 and is_diff_lane(scheme[i + 1], scheme[i], list1):
            # 有后车
            d2 = max(min((a_time[scheme[i + 1]] - a_time[scheme[i]]) / 3, 1), 0.5)
            ans_utility[scheme[i]] += d2 / 2 * utility[scheme[i]] + (1 - d2) / 2 * utility[scheme[i + 1]]
        else:
            ans_utility[scheme[i]] += utility[scheme[i]] / 2
    return ans_utility, timetable


def processB_hv(scheme_A, T_A_scheme, U_A_scheme, B1, B2, standard_timetable_A, hvlist):
    all_scheme_AB = []
    U_B_scheme, T_B_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Bi, U_Bi_scheme, T_Bi_scheme = processBi_hv(T_A_scheme[scheme_i], B1, B2, standard_timetable_A, hvlist)
        for k in scheme_Bi:
            all_scheme_AB.append((scheme_i, k))
        U_B_scheme[scheme_i] = U_Bi_scheme
        T_B_scheme[scheme_i] = T_Bi_scheme
    # all_scheme_B=set(all_scheme_B)
    S_U_AB = back_updateAB(all_scheme_AB, U_A_scheme, U_B_scheme, list(B1.keys()), list(B2.keys()))
    return S_U_AB, T_B_scheme

def processC_hv(scheme_A, T_A_scheme, U_A_scheme, C1, C2, standard_timetable_A, hvlist):
    all_scheme_AC = []
    U_C_scheme, T_C_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Ci, U_Ci_scheme, T_Ci_scheme = processCi_hv(T_A_scheme[scheme_i], C1, C2, standard_timetable_A, hvlist)
        for k in scheme_Ci:
            all_scheme_AC.append((scheme_i, k))
        U_C_scheme[scheme_i] = U_Ci_scheme
        T_C_scheme[scheme_i] = T_Ci_scheme
    # all_scheme_B=set(all_scheme_B)
    S_U_AC = back_updateAC(all_scheme_AC, U_A_scheme, U_C_scheme, list(C1.keys()), list(C2.keys()))
    return S_U_AC, T_C_scheme


def processBi(T_A_scheme_i, B1, B2, standard_timetable_A):
    B1_time = {i: B1[i][0] / (B1[i][1] + 0.01) for i in B1.keys()}
    B2_time, B2i_time = {}, {}
    for i in B2.keys():
        if i in standard_timetable_A.keys():
            B2_time[i] = standard_timetable_A[i] + 6 / (B2[i][1] + 0.01)
            B2i_time[i] = T_A_scheme_i[i] + 6 / (B2[i][1] + 0.01)
        else:
            B2_time[i] = B2[i][0] / (B2[i][1] + 0.01)
            B2i_time[i] = B2[i][0] / (B2[i][1] + 0.01)

    # B2_time = {i: B2[i][0] / (B2[i][1] + 0.01) for i in B2.keys()}
    # B2i_time = {i: T_A_scheme_i[i] + 6/(B2[i][1] + 0.01) for i in B2.keys()}

    B_time = {**B1_time, **B2_time}
    Bi_time = {**B1_time, **B2i_time}

    B1_id = sorted(B1_time, key=lambda k: B1_time[k])
    B2_id = sorted(B2_time, key=lambda k: B2_time[k])
    B2i_id = sorted(B2i_time, key=lambda k: B2i_time[k])

    B_id = sorted(B_time, key=lambda k: B_time[k])
    Bi_id = sorted(Bi_time, key=lambda k: Bi_time[k])

    standard_scheme_B = deepcopy(B_id)  # 默认先到先得顺序
    standard_scheme_Bi = deepcopy(Bi_id)  # 在A顺序条件下的先到先得顺序
    standard_timetable_B = generate_timetable(standard_scheme_B, B_time)  # 默认的在FIFS顺序下的到达时间
    standard_timetable_Bi = generate_timetable(standard_scheme_Bi, Bi_time)  # 在A顺序条件下的在FIFS顺序下的到达时间
    scheme_Bi = generate_permutations(B1_id, B2_id)  # 所有可能顺序

    U_Bi_scheme, T_Bi_scheme = {}, {}
    for scheme_Bi_j in scheme_Bi:
        utility, timetable = pruner(scheme_Bi_j, Bi_time, standard_timetable_B, B1_id, 3)
        if utility is not None:
            U_Bi_scheme[tuple(scheme_Bi_j)] = utility
            T_Bi_scheme[tuple(scheme_Bi_j)] = timetable
    if len(U_Bi_scheme) == 0:
        return [], U_Bi_scheme, T_Bi_scheme
    else:
        scheme_Bi = list(U_Bi_scheme.keys())
        return scheme_Bi, U_Bi_scheme, T_Bi_scheme


def processBi_hv(T_A_scheme_i, B1, B2, standard_timetable_A, hvlist):
    B1_time = {i: B1[i][0] / (B1[i][1] + 0.01) for i in B1.keys()}
    B2_time, B2i_time = {}, {}
    for i in B2.keys():
        if i in standard_timetable_A.keys():
            B2_time[i] = standard_timetable_A[i] + 6 / (B2[i][1] + 0.01)
            B2i_time[i] = T_A_scheme_i[i] + 6 / (B2[i][1] + 0.01)
        else:
            B2_time[i] = B2[i][0] / (B2[i][1] + 0.01)
            B2i_time[i] = B2[i][0] / (B2[i][1] + 0.01)

    # B2_time = {i: B2[i][0] / (B2[i][1] + 0.01) for i in B2.keys()}
    # B2i_time = {i: T_A_scheme_i[i] + 6/(B2[i][1] + 0.01) for i in B2.keys()}

    B_time = {**B1_time, **B2_time}
    Bi_time = {**B1_time, **B2i_time}

    B1_id = sorted(B1_time, key=lambda k: B1_time[k])
    B2_id = sorted(B2_time, key=lambda k: B2_time[k])
    B2i_id = sorted(B2i_time, key=lambda k: B2i_time[k])

    B_id = sorted(B_time, key=lambda k: B_time[k])
    Bi_id = sorted(Bi_time, key=lambda k: Bi_time[k])

    hv1_id = [i for i in B1_id if i in hvlist]
    hv2_id = [i for i in B2_id if i in hvlist]

    standard_scheme_B = deepcopy(B_id)  # 默认先到先得顺序
    standard_scheme_Bi = deepcopy(Bi_id)  # 在A顺序条件下的先到先得顺序
    standard_timetable_B = generate_timetable(standard_scheme_B, B_time, hvlist)  # 默认的在FIFS顺序下的到达时间
    standard_timetable_Bi = generate_timetable(standard_scheme_Bi, Bi_time, hvlist)  # 在A顺序条件下的在FIFS顺序下的到达时间
    scheme_Bi = generate_permutations(B1_id, B2_id)  # 所有可能顺序
    # 剔除不符合hv要求的顺序
    wrong_scheme = []
    for s in scheme_Bi:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于3.36s，则认为方案不行
            p = s.index(vid)
            if p != 0 and s[p - 1] in B2_id and Bi_time[s[p - 1]] - Bi_time[vid] < 3.36:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于1s，则认为不行
            p = s.index(vid)
            if p != 0 and p != len(s) - 1 and s[p - 1] in B1_id and s[p + 1] in B1_id and Bi_time[s[p + 1]] - Bi_time[
                s[p - 1]] < 1:
                wrong_scheme.append(s)
                break
    for s in wrong_scheme:
        scheme_Bi.remove(s)
    U_Bi_scheme, T_Bi_scheme = {}, {}
    for scheme_Bi_j in scheme_Bi:
        utility, timetable = pruner(scheme_Bi_j, Bi_time, standard_timetable_B, B1_id, 3)
        if utility is not None:
            U_Bi_scheme[tuple(scheme_Bi_j)] = utility
            T_Bi_scheme[tuple(scheme_Bi_j)] = timetable
    if len(U_Bi_scheme) == 0:
        return [], U_Bi_scheme, T_Bi_scheme
    else:
        scheme_Bi = list(U_Bi_scheme.keys())
        return scheme_Bi, U_Bi_scheme, T_Bi_scheme


def back_updateAB(all_scheme_AB, U_A_scheme, U_B_scheme, B1id, B2id):
    # 目的是获得A各方案的效用，B各方案的效用
    U_AB_scheme = {}
    if len(all_scheme_AB) == 0:
        all_id = []
    else:
        all_id = list({*list(all_scheme_AB[0][0]), *list(all_scheme_AB[0][1])})
    for s in all_scheme_AB:
        U_scheme = {}
        for vid in all_id:
            if vid in B1id:
                U_scheme[vid] = U_B_scheme[s[0]][s[1]][vid]
            elif vid in B2id:
                if vid in U_A_scheme[s[0]].keys():
                    U_scheme[vid] = U_B_scheme[s[0]][s[1]][vid] * U_A_scheme[s[0]][vid]
                else:
                    U_scheme[vid] = U_B_scheme[s[0]][s[1]][vid]
            else:
                U_scheme[vid] = U_A_scheme[s[0]][vid]
        U_AB_scheme[s] = U_scheme
    return U_AB_scheme


def processCi(T_A_scheme_i, C1, C2, standard_timetable_A):
    C2_time = {i: C2[i][0] / (C2[i][1] + 0.01) for i in C2.keys()}
    C1_time, C1i_time = {}, {}
    for i in C1.keys():
        if i in standard_timetable_A.keys():
            C1_time[i] = standard_timetable_A[i] + 6 / (C1[i][1] + 0.01)
            C1i_time[i] = T_A_scheme_i[i] + 6 / (C1[i][1] + 0.01)
        else:
            C1_time[i] = C1[i][0] / (C1[i][1] + 0.01)
            C1i_time[i] = C1[i][0] / (C1[i][1] + 0.01)

    # C1_time = {i: C1[i][0] / (C1[i][1] + 0.01) for i in C1.keys()}

    C_time = {**C1_time, **C2_time}
    Ci_time = {**C1i_time, **C2_time}

    C1_id = sorted(C1_time, key=lambda k: C1_time[k])
    C2_id = sorted(C2_time, key=lambda k: C2_time[k])
    C1i_id = sorted(C1i_time, key=lambda k: C1i_time[k])

    C_id = sorted(C_time, key=lambda k: C_time[k])
    Ci_id = sorted(Ci_time, key=lambda k: Ci_time[k])

    standard_scheme_C = deepcopy(C_id)  # 默认先到先得顺序
    standard_scheme_Ci = deepcopy(Ci_id)  # 在A顺序条件下的先到先得顺序
    standard_timetable_C = generate_timetable(standard_scheme_C, C_time)  # 默认的在FIFS顺序下的到达时间
    standard_timetable_Ci = generate_timetable(standard_scheme_Ci, Ci_time)  # 在A顺序条件下的在FIFS顺序下的到达时间
    scheme_Ci = generate_permutations(C1_id, C2_id)  # 所有可能顺序

    U_Ci_scheme, T_Ci_scheme = {}, {}
    for scheme_Ci_j in scheme_Ci:
        utility, timetable = pruner(scheme_Ci_j, Ci_time, standard_timetable_C, C1_id, 3)
        if utility is not None:
            U_Ci_scheme[tuple(scheme_Ci_j)] = utility
            T_Ci_scheme[tuple(scheme_Ci_j)] = timetable
    if len(U_Ci_scheme) == 0:
        return [], U_Ci_scheme, T_Ci_scheme
    else:
        scheme_Ci = list(U_Ci_scheme.keys())
        return scheme_Ci, U_Ci_scheme, T_Ci_scheme


def processCi_hv(T_A_scheme_i, C1, C2, standard_timetable_A, hvlist):
    C2_time = {i: C2[i][0] / (C2[i][1] + 0.01) for i in C2.keys()}
    C1_time, C1i_time = {}, {}
    for i in C1.keys():
        if i in standard_timetable_A.keys():
            C1_time[i] = standard_timetable_A[i] + 6 / (C1[i][1] + 0.01)
            C1i_time[i] = T_A_scheme_i[i] + 6 / (C1[i][1] + 0.01)
        else:
            C1_time[i] = C1[i][0] / (C1[i][1] + 0.01)
            C1i_time[i] = C1[i][0] / (C1[i][1] + 0.01)

    # C1_time = {i: C1[i][0] / (C1[i][1] + 0.01) for i in C1.keys()}

    C_time = {**C1_time, **C2_time}
    Ci_time = {**C1i_time, **C2_time}

    C1_id = sorted(C1_time, key=lambda k: C1_time[k])
    C2_id = sorted(C2_time, key=lambda k: C2_time[k])
    C1i_id = sorted(C1i_time, key=lambda k: C1i_time[k])

    C_id = sorted(C_time, key=lambda k: C_time[k])
    Ci_id = sorted(Ci_time, key=lambda k: Ci_time[k])

    hv1_id = [i for i in C1_id if i in hvlist]
    hv2_id = [i for i in C2_id if i in hvlist]

    standard_scheme_C = deepcopy(C_id)  # 默认先到先得顺序
    standard_scheme_Ci = deepcopy(Ci_id)  # 在A顺序条件下的先到先得顺序
    standard_timetable_C = generate_timetable(standard_scheme_C, C_time, hvlist)  # 默认的在FIFS顺序下的到达时间
    standard_timetable_Ci = generate_timetable(standard_scheme_Ci, Ci_time, hvlist)  # 在A顺序条件下的在FIFS顺序下的到达时间
    scheme_Ci = generate_permutations(C1_id, C2_id)  # 所有可能顺序
    # 剔除不符合hv要求的顺序
    wrong_scheme = []
    for s in scheme_Ci:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于3.36s，则认为方案不行
            p = s.index(vid)
            if p != 0 and s[p - 1] in C2_id and Ci_time[s[p - 1]] - Ci_time[vid] < 3.36:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于3.36s，则认为不行
            p = s.index(vid)
            if p != 0 and p != len(s) - 1 and s[p - 1] in C1_id and s[p + 1] in C1_id and Ci_time[s[p + 1]] - Ci_time[
                s[p - 1]] < 3.36:
                wrong_scheme.append(s)
                break
    for s in wrong_scheme:
        scheme_Ci.remove(s)
    U_Ci_scheme, T_Ci_scheme = {}, {}
    for scheme_Ci_j in scheme_Ci:
        utility, timetable = pruner(scheme_Ci_j, Ci_time, standard_timetable_C, C1_id, 3)
        if utility is not None:
            U_Ci_scheme[tuple(scheme_Ci_j)] = utility
            T_Ci_scheme[tuple(scheme_Ci_j)] = timetable
    if len(U_Ci_scheme) == 0:
        return [], U_Ci_scheme, T_Ci_scheme
    else:
        scheme_Ci = list(U_Ci_scheme.keys())
        return scheme_Ci, U_Ci_scheme, T_Ci_scheme

def back_updateAC(all_scheme_AC, U_A_scheme, U_C_scheme, C1id, C2id):
# 目的是获得A各方案的效用，C各方案的效用
    U_AC_scheme = {}
    if len(all_scheme_AC) == 0:
        all_id = []
    else:
        all_id = list({*list(all_scheme_AC[0][0]), *list(all_scheme_AC[0][1])})
    for s in all_scheme_AC:
        U_scheme = {}
        for vid in all_id:
            if vid in C1id:
                if vid in U_A_scheme[s[0]].keys():
                    U_scheme[vid] = U_C_scheme[s[0]][s[1]][vid] * U_A_scheme[s[0]][vid]
                else:
                    U_scheme[vid] = U_C_scheme[s[0]][s[1]][vid]
            elif vid in C2id:
                U_scheme[vid] = U_C_scheme[s[0]][s[1]][vid]
            else:
                U_scheme[vid] = U_A_scheme[s[0]][vid]
        U_AC_scheme[s] = U_scheme
    return U_AC_scheme


A1={3: [16.624157024086603, 5, 0], 5: [26.82951177170079, 5, 0]}
A2={2: [15.748946414924074, 5, 0]}
B1={4: [9.622736635752112, 5, 0]}
B2={2: [24.391265191537798, 5, 0]}
C1={3: [25.833918023038244, 5, 0], 5: [36.03927277065243, 5, 0]}
C2={1: [26.925093786948594, 5, 0]}


#A1,B1,C1,A2,B2,C2=[{100001: [5, 8, -0.45]},{},{100001: [12, 8.1, 0]},{},{},{}]
B1id=list(B1.keys())
B2id=list(B2.keys())
C1id=list(C1.keys())
C2id=list(C2.keys())
hvlist = []
start_time = time.perf_counter()
scheme_A, U_A_scheme, T_A_scheme, standard_timetable_A = processA_hv(A1, A2, hvlist)
S_U_AB, T_B_scheme = processB_hv(scheme_A, T_A_scheme, U_A_scheme, B1, B2, standard_timetable_A, hvlist)
S_U_AC, T_C_scheme = processC_hv(scheme_A, T_A_scheme, U_A_scheme, C1, C2, standard_timetable_A, hvlist)
end_time = time.perf_counter()
elapsed_time = end_time - start_time


