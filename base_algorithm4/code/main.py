from copy import deepcopy
import numpy as np
import random

def commu_conse(iterated_probability_ABC, shapley, hvlist=[]):  # 主要内容
    scheme = tuple(iterated_probability_ABC.keys())
    if len(scheme) != 0:
        member = iterated_probability_ABC[scheme[0]].keys()
    else:
        member = []
    member_preference = {i: np.array([iterated_probability_ABC[k][i] for k in scheme]) for i in member}
    for _ in range(1):
        omega_all = {i: cal_entro(member_preference[i]) for i in member}
        prefer_new = deepcopy(omega_all)
        for m in member:
            p = member_preference[m]
            for k in member:
                if k != m:
                    if shapley is None:
                        p += 0.15 * omega_all[k]
                    else:
                        p += shapley[m][k] * omega_all[k]
                        # p += 0.15 * omega_all[k]
            prefer_new[m] = p / sum(p)
        member_preference = deepcopy(prefer_new)
    # 每个个体挑选自己偏好最大的那个方案，返回
    act_scheme = {k: scheme[random.sample(max_index(member_preference[k]), 1)[0]] for k in member}
    return member_preference, act_scheme

def cal_shapley(iterated_probability_ABC, filtered_S_U_ABC, hvlist):  # 计算边际效用
    # 用于计算条件边界贡献，输入为最终备选方案及个体概率，所有方案及效用
    utility = {i: filtered_S_U_ABC[i] for i in iterated_probability_ABC.keys()}
    if len(utility) == 0:
        allid = []
    else:
        allid = [i for i in list(utility.values())[0].keys() if i not in hvlist]
    allshap = {i: {j: 0.15 for j in allid if j != i} for i in allid}
    sumpro = {i: sum(iterated_probability_ABC[i].values()) for i in iterated_probability_ABC.keys()}
    allpro = {i: sumpro[i] / sum(sumpro.values()) for i in sumpro.keys()}
    for i in allid:
        v0 = sum([allpro[k] * utility[k][i] for k in allpro.keys()])
        for j in allshap[i].keys():
            # 计算j存在与否对i的收益的变化
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

def cal_entro(x):  # 计算熵
    answer = list(np.zeros_like(x))
    xx = np.array([i for i in x if i != 0])
    entro = -np.sum(xx * np.log(xx))
    omega = 1 / (1 + entro)
    pos = max_index(x)
    for i in pos:
        answer[i] = omega
    return np.array(answer)

def max_index(lst_int):  # 选择最大的熵对应的方案
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return index

iterated_probability_ABC = {((3, 2, 5), (4, 2), (3, 1, 5)): {4: 0.3333333333333333,
  2: 0.10616075702593158,
  3: 0.6393238515077168,
  5: 0.449570199174057,
  1: 0.38851634887080255},
 ((2, 3, 5), (4, 2), (3, 1, 5)): {4: 0.3333333333333333,
  2: 0.4469196214870342,
  3: 0.22156473403624216,
  5: 0.2657761207276358,
  1: 0.21740535484007797},
 ((2, 3, 5), (4, 2), (1, 3, 5)): {4: 0.3333333333333333,
  2: 0.4469196214870342,
  3: 0.139111414456041,
  5: 0.2846536800983071,
  1: 0.3940782962891195}}

filtered_S_U_ABC = {((3, 2, 5), (4, 2), (3, 1, 5)): {4: 1.0,
  2: 0.23753881441298827,
  3: 4.595768463771477,
  5: 1.5793584647098005,
  1: 0.9858861869057708},
 ((3, 2, 5), (4, 2), (1, 3, 5)): {4: 1.0,
  2: 0.23753881441298827,
  3: 1.5768779233770283,
  5: 0.9243987698277935,
  1: 1.0},
 ((2, 3, 5), (4, 2), (3, 1, 5)): {4: 1.0,
  2: 1.0,
  3: 1.5927142636182188,
  5: 0.933682363199549,
  1: 0.5516806098871692},
 ((2, 3, 5), (4, 2), (1, 3, 5)): {4: 1.0, 2: 1.0, 3: 1.0, 5: 1.0, 1: 1.0}}

shapley = cal_shapley(iterated_probability_ABC, filtered_S_U_ABC, hvlist=[])
member_preference, act_scheme = commu_conse(iterated_probability_ABC, shapley, hvlist=[])

