from copy import deepcopy
import numpy as np
import random

def process_ABC(S_U_AB, S_U_AC, B1id, B2id, C1id, C2id):  # 整合所有的方案
    vehid = [*B1id, *B2id, *C1id, *C2id]
    S_U_ABC = {}
    for i in S_U_AB.keys():
        for j in S_U_AC.keys():
            if i[0] != j[0]:
                continue
            else:
                scheme = (i[0], i[1], j[1])
                S_U_ABC[scheme] = {}
                for k in vehid:
                    if k in B1id or k in B2id:
                        S_U_ABC[scheme][k] = S_U_AB[i][k]
                    elif k in C1id or k in C2id:
                        S_U_ABC[scheme][k] = S_U_AC[j][k]
                    else:
                        print("ERROR!")
    return S_U_ABC


def dominated_elimination(S_U_ABC, hvlist=[]):  # 剔除劣势方案
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


def prob_ABC(filtered_S_U_ABC, hvlist):  # 根据效用计算概率
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


filtered_S_U_ABC= {((3, 2, 5), (4, 2), (3, 1, 5)): {4: 1.0,
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

probability_ABC = prob_ABC(filtered_S_U_ABC, hvlist=[])
iterated_probability_ABC = iterated_select(probability_ABC, hvlist=[])

