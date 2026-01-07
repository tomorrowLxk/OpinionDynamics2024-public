# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:15:40 2024

@author: tomorrow
"""
from copy import deepcopy
import numpy as np
import random
import casadi as ca
def generate_permutations(list1, list2):
    # 用于求解全排列
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


def is_diff_lane(id1, id2, lane_list):
    if id1 in lane_list and id2 not in lane_list:
        return True
    elif id1 not in lane_list and id2 in lane_list:
        return True
    else:
        return False


def pruner(scheme: list, a_time: dict, std_timetable: dict, list1: list, t_res=4.5, hvlist=[]):
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
            utility[vid] = np.exp((t2 - t1)/2)
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


def generate_timetable(scheme: list, a_time: dict, hvlist=[]) -> dict:
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
                timetable[vehid] = max(a_time[vehid]*0.9, min(a_time[vehid], 1))
        else:
            if scheme[i - 1] in hvlist:
                timetable[vehid] = max(timetable[scheme[i - 1]] + 1.8, a_time[vehid])
            else:
                timetable[vehid] = max(timetable[scheme[i - 1]] + 1.5, a_time[vehid]*0.9)
    return timetable


def processBi(T_A_scheme_i, B1, B2, standard_timetable_A):
    B1_time = {i: B1[i][0] / (B1[i][1] + 0.01) for i in B1.keys()}    
    B2_time, B2i_time = {}, {}
    for i in B2.keys():
        if i in standard_timetable_A.keys():
            B2_time[i]=standard_timetable_A[i]+6/ (B2[i][1] + 0.01)
            B2i_time[i]=T_A_scheme_i[i] + 6/(B2[i][1] + 0.01)
        else:
            B2_time[i]=B2[i][0] / (B2[i][1] + 0.01)
            B2i_time[i]=B2[i][0] / (B2[i][1] + 0.01)
        
    #B2_time = {i: B2[i][0] / (B2[i][1] + 0.01) for i in B2.keys()}
    #B2i_time = {i: T_A_scheme_i[i] + 6/(B2[i][1] + 0.01) for i in B2.keys()}
    
    
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
        scheme_Bi=list(U_Bi_scheme.keys())
        return scheme_Bi, U_Bi_scheme, T_Bi_scheme


def processBi_hv(T_A_scheme_i, B1, B2, standard_timetable_A, hvlist):
    B1_time = {i: B1[i][0] / (B1[i][1] + 0.01) for i in B1.keys()}    
    B2_time, B2i_time = {}, {}
    for i in B2.keys():
        if i in standard_timetable_A.keys():
            B2_time[i]=standard_timetable_A[i]+6/ (B2[i][1] + 0.01)
            B2i_time[i]=T_A_scheme_i[i] + 6/(B2[i][1] + 0.01)
        else:
            B2_time[i]=B2[i][0] / (B2[i][1] + 0.01)
            B2i_time[i]=B2[i][0] / (B2[i][1] + 0.01)
        
    #B2_time = {i: B2[i][0] / (B2[i][1] + 0.01) for i in B2.keys()}
    #B2i_time = {i: T_A_scheme_i[i] + 6/(B2[i][1] + 0.01) for i in B2.keys()}
    
    
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
    wrong_scheme=[]
    for s in scheme_Bi:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于0.5s，则认为方案不行
            p=s.index(vid)
            if p!=0 and s[p-1] in B2_id and Bi_time[s[p-1]]-Bi_time[vid]<0.5:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于1s，则认为不行
            p = s.index(vid)
            if p!=0 and p!=len(s)-1 and s[p-1] in B1_id and s[p+1] in B1_id and Bi_time[s[p+1]]-Bi_time[s[p-1]]<1:
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
        scheme_Bi=list(U_Bi_scheme.keys())
        return scheme_Bi, U_Bi_scheme, T_Bi_scheme


def back_updateAB(all_scheme_AB, U_A_scheme, U_B_scheme, B1id, B2id):
    # 目的是获得A各方案的效用，B各方案的效用
    U_AB_scheme={}
    all_id = list({*list(all_scheme_AB[0][0]), *list(all_scheme_AB[0][1])})
    for s in all_scheme_AB:
        U_scheme={}
        for vid in all_id:
            if vid in B1id:
                U_scheme[vid]=U_B_scheme[s[0]][s[1]][vid]
            elif vid in B2id:
                if vid in U_A_scheme[s[0]].keys():
                    U_scheme[vid]=U_B_scheme[s[0]][s[1]][vid]*U_A_scheme[s[0]][vid]
                else:
                    U_scheme[vid]=U_B_scheme[s[0]][s[1]][vid]
            else:
                U_scheme[vid]=U_A_scheme[s[0]][vid]
        U_AB_scheme[s]=U_scheme     
    return U_AB_scheme


def processCi(T_A_scheme_i, C1, C2, standard_timetable_A):
    C2_time = {i: C2[i][0] / (C2[i][1] + 0.01) for i in C2.keys()}  
    C1_time, C1i_time = {}, {}
    for i in C1.keys():
        if i in standard_timetable_A.keys():
            C1_time[i]=standard_timetable_A[i]+6/ (C1[i][1] + 0.01)
            C1i_time[i]=T_A_scheme_i[i] + 6/(C1[i][1] + 0.01)
        else:
            C1_time[i]=C1[i][0] / (C1[i][1] + 0.01)
            C1i_time[i]=C1[i][0] / (C1[i][1] + 0.01)
    
    #C1_time = {i: C1[i][0] / (C1[i][1] + 0.01) for i in C1.keys()}
    
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
        scheme_Ci=list(U_Ci_scheme.keys())
        return scheme_Ci, U_Ci_scheme, T_Ci_scheme
    

def processCi_hv(T_A_scheme_i, C1, C2, standard_timetable_A, hvlist):
    C2_time = {i: C2[i][0] / (C2[i][1] + 0.01) for i in C2.keys()}  
    C1_time, C1i_time = {}, {}
    for i in C1.keys():
        if i in standard_timetable_A.keys():
            C1_time[i]=standard_timetable_A[i]+6/ (C1[i][1] + 0.01)
            C1i_time[i]=T_A_scheme_i[i] + 6/(C1[i][1] + 0.01)
        else:
            C1_time[i]=C1[i][0] / (C1[i][1] + 0.01)
            C1i_time[i]=C1[i][0] / (C1[i][1] + 0.01)
    
    #C1_time = {i: C1[i][0] / (C1[i][1] + 0.01) for i in C1.keys()}
    
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
    wrong_scheme=[]
    for s in scheme_Ci:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于0.5s，则认为方案不行
            p=s.index(vid)
            if p!=0 and s[p-1] in C2_id and Ci_time[s[p-1]]-Ci_time[vid]<0.5:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于1s，则认为不行
            p = s.index(vid)
            if p!=0 and p!=len(s)-1 and s[p-1] in C1_id and s[p+1] in C1_id and Ci_time[s[p+1]]-Ci_time[s[p-1]]<1:
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
        scheme_Ci=list(U_Ci_scheme.keys())
        return scheme_Ci, U_Ci_scheme, T_Ci_scheme


def back_updateAC(all_scheme_AC, U_A_scheme, U_C_scheme, C1id, C2id):
    # 目的是获得A各方案的效用，C各方案的效用
    U_AC_scheme={}
    all_id = list({*list(all_scheme_AC[0][0]), *list(all_scheme_AC[0][1])})
    for s in all_scheme_AC:
        U_scheme={}
        for vid in all_id:
            if vid in C1id:
                if vid in U_A_scheme[s[0]].keys():
                    U_scheme[vid]=U_C_scheme[s[0]][s[1]][vid]*U_A_scheme[s[0]][vid]
                else:
                    U_scheme[vid]=U_C_scheme[s[0]][s[1]][vid]
            elif vid in C2id:
                U_scheme[vid]=U_C_scheme[s[0]][s[1]][vid]
            else:
                U_scheme[vid]=U_A_scheme[s[0]][vid]
        U_AC_scheme[s]=U_scheme        
    return U_AC_scheme


def is_subsequence_continuous(old_seq, new_seq):
    # 判断子序列连续
    common_elements = [elem for elem in old_seq if elem in new_seq]
    if len(common_elements) <= 1:
        return True
    start_index = 0
    for elem in common_elements:
        try:
            current_index = new_seq.index(elem, start_index)
        except ValueError:
            return False
        if current_index != start_index:
            return False
        start_index += 1
    return True


def are_plans_continuous(old_plan, new_plan):
    # 判断两方案相同
    for old_seq, new_seq in zip(old_plan, new_plan):
        if not is_subsequence_continuous(old_seq, new_seq):
            return False
    return True


'''
A1={2:[3,5,0],4:[11,5,0]}
A2={6:[8,5,0],7:[14,5,0]}
B1={1:[3,5,0],3:[13,5,0]}
B2={6:[14,5,0],7:[20,5,0]}
C1={2:[9,5,0],4:[17,5,0]}
C2={5:[6,5,0],8:[17,5,0]}
'''
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


def processA(A1, A2):
    A1_time = {i: A1[i][0] / (A1[i][1] + 0.01) for i in A1.keys()}
    A2_time = {i: A2[i][0] / (A2[i][1] + 0.01) for i in A2.keys()}
    A_time = {**A1_time, **A2_time}
    A1_id = sorted(A1_time, key=lambda k: A1_time[k])
    A2_id = sorted(A2_time, key=lambda k: A2_time[k])
    A_id = sorted(A_time, key=lambda k: A_time[k])
    standard_scheme_A = deepcopy(A_id)  # 先到先得顺序
    standard_timetable_A = generate_timetable(standard_scheme_A, A_time)  # 记录在FIFS顺序下的到达时间
    all_scheme_A = generate_permutations(A1_id, A2_id)  # 所有可能顺序
    U_A_scheme, T_A_scheme = {}, {}
    for scheme_i in all_scheme_A:
        utility, timetable = pruner(scheme_i, A_time, standard_timetable_A, A1_id)
        if utility is not None:
            U_A_scheme[tuple(scheme_i)] = utility
            T_A_scheme[tuple(scheme_i)] = timetable
    if list(T_A_scheme.keys())[0] == ():
        return [()], U_A_scheme, T_A_scheme, standard_timetable_A
    else:  
        scheme_A=list(U_A_scheme.keys())
        #P_A_scheme = {sc: {vehid: U_A_scheme[sc][vehid] / sum(i[vehid] for i in U_A_scheme.values()) for vehid in A_id} for sc in scheme_A}
        return scheme_A, U_A_scheme, T_A_scheme, standard_timetable_A


def processA_hv(A1, A2, hvlist):
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
    wrong_scheme=[]
    for s in all_scheme_A:
        for vid in hv1_id:
            # 环岛内部HV，如果有前车且前车是环岛外车且环岛外车领先时间小于0.5s，则认为方案不行
            p=s.index(vid)
            if p!=0 and s[p-1] in A2_id and A_time[s[p-1]]-A_time[vid]<0.5:
                wrong_scheme.append(s)
                break
        if s in wrong_scheme:
            continue
        for vid in hv2_id:
            # 环岛外部HV，如果有前车与后车，且前车与后车都是环岛内部车辆，且前车与后车时间差小于1s，则认为不行
            p = s.index(vid)
            if p!=0 and p!=len(s)-1 and s[p-1] in A1_id and s[p+1] in A1_id and A_time[s[p+1]]-A_time[s[p-1]]<1:
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
        return [()], U_A_scheme, T_A_scheme, standard_timetable_A
    else:
        scheme_A=list(U_A_scheme.keys())
        #P_A_scheme = {sc: {vehid: U_A_scheme[sc][vehid] / sum(i[vehid] for i in U_A_scheme.values()) for vehid in A_id} for sc in scheme_A}
        return scheme_A, U_A_scheme, T_A_scheme, standard_timetable_A


def processB(scheme_A, T_A_scheme, U_A_scheme, B1, B2, standard_timetable_A):
    all_scheme_AB=[]
    U_B_scheme, T_B_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Bi, U_Bi_scheme, T_Bi_scheme = processBi(T_A_scheme[scheme_i], B1, B2, standard_timetable_A)
        for k in scheme_Bi:
            all_scheme_AB.append((scheme_i,k))
        U_B_scheme[scheme_i]=U_Bi_scheme
        T_B_scheme[scheme_i]=T_Bi_scheme
    #all_scheme_B=set(all_scheme_B)
    S_U_AB=back_updateAB(all_scheme_AB, U_A_scheme, U_B_scheme, list(B1.keys()), list(B2.keys()))
    return S_U_AB, T_B_scheme
    
    
def processB_hv(scheme_A, T_A_scheme, U_A_scheme, B1, B2, standard_timetable_A, hvlist):
    all_scheme_AB=[]
    U_B_scheme, T_B_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Bi, U_Bi_scheme, T_Bi_scheme = processBi_hv(T_A_scheme[scheme_i], B1, B2, standard_timetable_A, hvlist)
        for k in scheme_Bi:
            all_scheme_AB.append((scheme_i,k))
        U_B_scheme[scheme_i]=U_Bi_scheme
        T_B_scheme[scheme_i]=T_Bi_scheme
    #all_scheme_B=set(all_scheme_B)
    S_U_AB=back_updateAB(all_scheme_AB, U_A_scheme, U_B_scheme, list(B1.keys()), list(B2.keys()))
    return S_U_AB, T_B_scheme   
    
    
def processC(scheme_A, T_A_scheme, U_A_scheme, C1, C2, standard_timetable_A):
    all_scheme_AC=[]
    U_C_scheme, T_C_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Ci, U_Ci_scheme, T_Ci_scheme = processCi(T_A_scheme[scheme_i], C1, C2, standard_timetable_A)
        for k in scheme_Ci:
            all_scheme_AC.append((scheme_i, k))
        U_C_scheme[scheme_i]=U_Ci_scheme
        T_C_scheme[scheme_i]=T_Ci_scheme
    #all_scheme_B=set(all_scheme_B)
    S_U_AC=back_updateAC(all_scheme_AC, U_A_scheme, U_C_scheme, list(C1.keys()), list(C2.keys()))
    return S_U_AC, T_C_scheme

    
def processC_hv(scheme_A, T_A_scheme, U_A_scheme, C1, C2, standard_timetable_A, hvlist):
    all_scheme_AC=[]
    U_C_scheme, T_C_scheme = {}, {}
    for scheme_i in scheme_A:
        scheme_Ci, U_Ci_scheme, T_Ci_scheme = processCi_hv(T_A_scheme[scheme_i], C1, C2, standard_timetable_A, hvlist)
        for k in scheme_Ci:
            all_scheme_AC.append((scheme_i, k))
        U_C_scheme[scheme_i]=U_Ci_scheme
        T_C_scheme[scheme_i]=T_Ci_scheme
    #all_scheme_B=set(all_scheme_B)
    S_U_AC=back_updateAC(all_scheme_AC, U_A_scheme, U_C_scheme, list(C1.keys()), list(C2.keys()))
    return S_U_AC, T_C_scheme


def process_ABC(S_U_AB, S_U_AC, B1id, B2id, C1id, C2id):
    '''
    
    Parameters
    ----------
    S_U_AB : dict
        由AB两点通行方案与各车效用构成的字典
    S_U_AC : dict
        由AC两点通行方案与各车效用构成的字典

    Returns
    -------
    dict
        ABC3点所有的通行方案、所有车对所有方案的效用

    '''
    vehid = [*B1id,*B2id,*C1id,*C2id]
    S_U_ABC={}
    for i in S_U_AB.keys():
        for j in S_U_AC.keys():
            if i[0]!=j[0]:
                continue
            else:
                scheme=(i[0],i[1],j[1])
                S_U_ABC[scheme]={}
                for k in vehid:
                    if k in B1id or k in B2id:
                        S_U_ABC[scheme][k]=S_U_AB[i][k]
                    elif k in C1id or k in C2id:
                        S_U_ABC[scheme][k]=S_U_AC[j][k]
                    else:
                        print("ERROR!")
    return S_U_ABC


def dominated_elimination(S_U_ABC, hvlist=[]):
    # 剔除严格劣势策略
    dominated_schemes = set()
    for scheme1 in S_U_ABC.keys():
        for scheme2 in S_U_ABC.keys():
            if scheme1!=scheme2:
                if all(S_U_ABC[scheme1][player] <= S_U_ABC[scheme2][player] for player in S_U_ABC[scheme1].keys() if player not in hvlist):
                    dominated_schemes.add(scheme1)
                    break
    filtered_S_U_ABC = {scheme: S_U_ABC[scheme] for scheme in S_U_ABC.keys() if scheme not in dominated_schemes}
    return filtered_S_U_ABC


def merged_scheme(old_S_U_ABC, filtered_S_U_ABC):
    # 融合两时刻的方案的效用
    merged_S_U_ABC=deepcopy(filtered_S_U_ABC)
    for i in old_S_U_ABC.keys():
        for j in filtered_S_U_ABC.keys():
            if are_plans_continuous(i, j):
                for k in merged_S_U_ABC[j].keys():
                    if k in old_S_U_ABC[i].keys():
                        merged_S_U_ABC[j][k]=np.sqrt(old_S_U_ABC[i][k]*filtered_S_U_ABC[j][k])                       
    return merged_S_U_ABC


def prob_ABC(filtered_S_U_ABC, hvlist):
    probability_ABC = {}
    player_total_rewards = {}
    for scheme in filtered_S_U_ABC.keys():
        for player, reward in filtered_S_U_ABC[scheme].items():
            if player not in hvlist:
                player_total_rewards[player] = player_total_rewards.get(player, 0) + reward

    # 计算每个玩家在每个方案中的选择概率
    for scheme in filtered_S_U_ABC.keys():
        for player, reward in filtered_S_U_ABC[scheme].items():
            if player not in hvlist:
                probability = reward / player_total_rewards[player]
                probability_ABC.setdefault(scheme, {})[player] = probability
    return probability_ABC


def iterated_select(probability_ABC, hvlist):
    flag=1
    iterated_S_U_ABC = deepcopy(probability_ABC)
    while flag:
        iterated=set()
        for scheme in iterated_S_U_ABC.keys():
            if sum(iterated_S_U_ABC[scheme].values())/len(iterated_S_U_ABC[scheme])<0.975/len(iterated_S_U_ABC) and min(iterated_S_U_ABC[scheme].values())<0.485/len(iterated_S_U_ABC):
                iterated.add(scheme)
        if len(iterated)==0:
            break
        iterated_S_U_ABC={scheme: iterated_S_U_ABC[scheme] for scheme in iterated_S_U_ABC.keys() if scheme not in iterated}
        iterated_S_U_ABC=prob_ABC(iterated_S_U_ABC, hvlist)
        if len(iterated_S_U_ABC)<=len(list(iterated_S_U_ABC.values())[0]):
            break
    iterated_probability_ABC=iterated_S_U_ABC
    return iterated_probability_ABC


def commu_conse(iterated_probability_ABC, shapley, hvlist=[]):
    scheme=tuple(iterated_probability_ABC.keys())
    if len(scheme)!=0:
        member = iterated_probability_ABC[scheme[0]].keys()
    else:
        member=[]
    member_preference={i:np.array([iterated_probability_ABC[k][i] for k in scheme]) for i in member}
    for _ in range(2):
        omega_all={i:cal_entro(member_preference[i]) for i in member}
        prefer_new = deepcopy(omega_all)
        for m in member:
            p=member_preference[m]
            for k in member:
                if k!=m:
                    if shapley==None:
                        p+=0.15*omega_all[k]
                    else:
                        p+=shapley[m][k]*omega_all[k]
            prefer_new[m]=p/sum(p)
        member_preference=deepcopy(prefer_new)
    # 每个个体挑选自己偏好最大的那个方案，返回
    act_scheme={k: scheme[random.sample(max_index(member_preference[k]),1)[0]] for k in member}
    return member_preference, act_scheme


def cal_shapley(iterated_probability_ABC, filtered_S_U_ABC, hvlist):
    # 用于计算条件边界贡献，输入为最终备选方案及个体概率，所有方案及效用
    utility={i:filtered_S_U_ABC[i] for i in iterated_probability_ABC.keys()}
    if len(utility) == 0:
        allid = []
    else:
        allid=[i for i in list(utility.values())[0].keys() if i not in hvlist]
    allshap = {i: {j: 0.15 for j in allid if j != i} for i in allid}
    sumpro = {i: sum(iterated_probability_ABC[i].values()) for i in iterated_probability_ABC.keys()}
    allpro = {i: sumpro[i]/sum(sumpro.values()) for i in sumpro.keys()}
    for i in allid:
        v0 = sum([allpro[k]*utility[k][i] for k in allpro.keys()])
        for j in allshap[i].keys():
            # 计算j存在与否对i的收益的变化
            sumpro2 = {k: sum(iterated_probability_ABC[k].values())-iterated_probability_ABC[k][j] for k in iterated_probability_ABC.keys()}
            allpro2 = {k: sumpro2[k]/sum(sumpro2.values()) for k in sumpro.keys()}
            v1 = sum([allpro2[k]*utility[k][i] for k in allpro2.keys()])
            allshap[i][j]=v0-v1
    allvalue=[j for i in allshap.values() for j in i.values()]
    if len(allvalue) == 0:
        return None
    maxvalue, minvalue = max(allvalue), min(allvalue)
    if maxvalue==minvalue:
        shapley={i:{j:0.15 for j in allid if j != i} for i in allid}
    else:
        shapley={i:{j:0.1+0.4*(allshap[i][j]-minvalue)/(maxvalue-minvalue) for j in allshap[i].keys()} for i in allshap.keys()}
    return shapley


def cal_entro(x):
    answer = list(np.zeros_like(x))
    xx = np.array([i for i in x if i != 0])
    entro = -np.sum(xx * np.log(xx))
    omega = 1 / (1 + entro)
    pos = max_index(x)
    for i in pos:
        answer[i]=omega
    return np.array(answer)


def max_index(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return index


def get_time(act_scheme, A1, A2, B1, B2, C1, C2, T_A_scheme, T_B_scheme, T_C_scheme, hvlist):
    fini_timewindow = {}
    for k in act_scheme.keys():
        if k in hvlist:
            continue
        fini_timewindow[k] = []
        s1, s2, s3 = act_scheme[k]
        if k in A1.keys():
            t1 = T_A_scheme[s1][k]
            d1 = A1[k][0]
            fini_timewindow[k].extend([t1, d1])
        if k in A2.keys():
            t1 = T_A_scheme[s1][k]
            d1 = A2[k][0]
            fini_timewindow[k].extend([t1, d1])
        if k in B1.keys():
            t2 = T_B_scheme[s1][s2][k]
            d2 = B1[k][0]
            fini_timewindow[k].extend([t2, d2])
        if k in B2.keys():
            t2 = T_B_scheme[s1][s2][k]
            d2 = B2[k][0]
            fini_timewindow[k].extend([t2, d2])
        if k in C1.keys():
            t3 = T_C_scheme[s1][s3][k]
            d3 = C1[k][0]
            fini_timewindow[k].extend([t3, d3])
        if k in C2.keys():
            t3 = T_C_scheme[s1][s3][k]
            d3 = C2[k][0]
            fini_timewindow[k].extend([t3, d3])
    return fini_timewindow


def get_a_v(A1, A2, B1, B2, C1, C2):
    a_v_table={}
    for i in [A1, A2, B1, B2, C1, C2]:
        for j in i.keys():
            a_v_table[j]=[i[j][1], i[j][2]]
    return a_v_table


'''
def commu_conse2(iterated_probability_ABC):
    scheme = tuple(iterated_probability_ABC.keys())
    member = iterated_probability_ABC[scheme[0]].keys()
    member_preference = {i: np.array([iterated_probability_ABC[k][i] for k in scheme]) for i in member}
    for _ in range(2):
        omega_all = {i: cal_entro(member_preference[i]) for i in member}
        omega_sum = sum(i for i in omega_all.values())
        prefer_new = deepcopy(omega_all)
        for m in member:
            prefer_new[m] = (member_preference[m] + 0.15 * omega_sum - 0.15 * omega_all[m]) / (
                    sum(member_preference[m]) + 0.15 * sum(omega_sum) - 0.15 * sum(omega_all[m]))
        member_preference = deepcopy(prefer_new)
    # 每个个体挑选自己偏好最大的那个方案，返回
    act_scheme = {k: scheme[random.sample(max_index(member_preference[k]), 1)[0]] for k in member}
    return member_preference, act_scheme
'''


def cal_shapley(iterated_probability_ABC, filtered_S_U_ABC, hvlist):
    # 用于计算条件边界贡献，输入为最终备选方案及个体概率，所有方案及效用
    # 有一个矩阵M，Mij是个体j对个体i的边际贡献
    # 有一个矩阵P，Nab是个体a选择方案b的可能性
    # 有一个矩阵F，Fab是个体a选择方案b的效用
    # 首先初始化M为0.15
    utility = {i: filtered_S_U_ABC[i] for i in iterated_probability_ABC.keys()}
    if len(utility) == 0:
        allid = []
    else:
        allid = [i for i in list(utility.values())[0].keys() if i not in hvlist]
    allshap = {i: {j: 0.15 for j in allid if j != i} for i in allid}
    # 然后将P的每一列相加
    sumpro = {i: sum(iterated_probability_ABC[i].values()) for i in iterated_probability_ABC.keys()}
    # 然后均一化
    allpro = {i: sumpro[i] / sum(sumpro.values()) for i in sumpro.keys()}
    # 针对CAV_i
    for i in allid:
        # 计算这个CAV的所有方案的效用期望值
        v0 = sum([allpro[k] * utility[k][i] for k in allpro.keys()])
        # 对于CAV_j
        for j in allshap[i].keys():
            # 计算j存在与否对i的收益的变化
            # 
            sumpro2 = {k: sum(iterated_probability_ABC[k].values()) - iterated_probability_ABC[k][j] for k in
                       iterated_probability_ABC.keys()}
            allpro2 = {k: sumpro2[k] / sum(sumpro2.values()) for k in sumpro.keys()}
            v1 = sum([allpro2[k] * utility[k][i] for k in allpro2.keys()])
            allshap[i][j] = v0 - v1
    allvalue = [j for i in allshap.values() for j in i.values()]
    if len(allvalue) == 0:
        return None
    maxvalue, minvalue = max(allvalue), min(allvalue)
    if maxvalue == minvalue:
        shapley = {i: {j: 0.15 for j in allid if j != i} for i in allid}
    else:
        shapley = {i: {j: 0.1 + 0.4 * (allshap[i][j] - minvalue) / (maxvalue - minvalue) for j in allshap[i].keys()} for
                   i in allshap.keys()}
    return shapley

def solve_mpc(v0, a0, t1, d1, t2=None, d2=None):
    # Define optimization parameters
    T = 20  # Time steps
    dt = 0.1  # Time step length
    N = T + 1  # Total number of nodes including the initial one

    # Define optimization variables
    aa = ca.SX.sym('aa', T)  # Jerk

    # Define initial state parameters
    P = ca.SX.sym('P', 3)  # Initial parameters: [a0, v0, s0]

    s = ca.SX.sym('s', N)  # position
    v = ca.SX.sym('v', N)  # velocity
    a = ca.SX.sym('a', N)  # acceleration

    # Initialize states
    a[0] = P[0]
    v[0] = P[1]
    s[0] = P[2]

    # Dynamics equations for states based on initial conditions and optimization variable
    for i in range(T):
        a[i + 1] = a[i] + aa[i] * dt
        v[i + 1] = v[i] + a[i] * dt
        s[i + 1] = s[i] + v[i] * dt + 0.5 * a[i] * dt ** 2

    # Constraints
    g = []
    lbg, ubg = [], []
    for i in range(1, N):
        g.append(v[i])  # 速度约束
        g.append(a[i])  # 加速度约束
        lbg.append(0)  # 速度下界
        ubg.append(15)  # 速度上界
        lbg.append(-5)  # 加速度下界
        ubg.append(3)  # 加速度上界

    lbx, ubx = [], []
    for _ in range(T):
        lbx.append(-30)
        ubx.append(30)

        # Objective function components
    cost_position = 0
    cost_velocity_deviation = ca.sum1(ca.sumsqr(v - 8))  # Deviation from target speed
    cost_jerk = ca.sum1(ca.sumsqr(aa))  # Minimize jerk

    # Handle terminal constraints and costs
    if t1 > T * dt:
        s_t1 = s[T] + (t1 - T * dt) * v[T] + 0.5 * a[T] * (t1 - T * dt) ** 2
        cost_position += (s_t1 - d1) ** 2
    else:
        cost_position += (s[int(t1 / dt + 0.5)] - d1) ** 2

    if t2 is not None:
        if t2 > T * dt:
            s_t2 = s[T] + (t2 - T * dt) * v[T] + 0.5 * a[T] * (t2 - T * dt) ** 2
            cost_position += (s_t2 - d2) ** 2
        else:
            cost_position += (s[int(t2 / dt + 0.5)] - d2) ** 2

    obj = cost_position + cost_jerk * 0.1 + 0.02 * cost_velocity_deviation

    # Create optimization problem
    nlp = {'x': ca.vertcat(aa), 'f': obj, 'g': ca.vertcat(*g), 'p': P}

    # Solver options
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-4}

    # Create and solve the NLPSolver
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)

    # Initial guess for 'aa'
    x0 = np.zeros(T)

    # Solve the optimization problem
    res = solver(x0=x0, p=np.array([min(max(a0, -5), 3), v0, 0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return res

import time 
hvlist=[]
start_time = time.perf_counter()
scheme_A, U_A_scheme, T_A_scheme, standard_timetable_A = processA_hv(A1, A2, hvlist)
S_U_AB, T_B_scheme = processB_hv(scheme_A, T_A_scheme, U_A_scheme, B1, B2, standard_timetable_A, hvlist)
S_U_AC, T_C_scheme = processC_hv(scheme_A, T_A_scheme, U_A_scheme, C1, C2, standard_timetable_A, hvlist)
S_U_ABC = process_ABC(S_U_AB, S_U_AC, B1id, B2id, C1id, C2id)
filtered_S_U_ABC = dominated_elimination(S_U_ABC, hvlist)
probability_ABC = prob_ABC(filtered_S_U_ABC, hvlist)
iterated_probability_ABC = iterated_select(probability_ABC, hvlist)
shapley = cal_shapley(iterated_probability_ABC, filtered_S_U_ABC, hvlist)
member_preference, act_scheme = commu_conse(iterated_probability_ABC, shapley, hvlist)
#member_preference2, act_scheme2 = commu_conse2(iterated_probability_ABC)

fini_timewindow = get_time(act_scheme, A1, A2, B1, B2, C1, C2, T_A_scheme, T_B_scheme, T_C_scheme, hvlist)
end_time = time.perf_counter()
a_v_table= get_a_v(A1, A2, B1, B2, C1, C2)

ans_acc = {}
for i in fini_timewindow.keys():
    start_time = time.perf_counter()
    v0, a0 = a_v_table[i]
    ft = fini_timewindow[i]
    if len(ft) == 4:
        res = solve_mpc(v0, a0, ft[0], ft[1], ft[2], ft[3])
    else:
        res = solve_mpc(v0, a0, ft[0], ft[1])
    aa = list(res['x'].full().flatten())
    ans_acc[i] = a0 + 0.1 * aa[0]
    end_time = time.perf_counter()
    print(end_time - start_time)

elapsed_time = end_time - start_time

