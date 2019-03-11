# -*- coding: utf-8 -*-
"""
Last modified on 20181018
@author: cqju
"""
import numpy as np

class Class_genpragh():
    def __init__(self, **kwargs):
        pass

    def generate_pragh_set(case_in):
        if "no_of_nodes" in case_in.keys():
            set_V = [dict() for i in range(case_in["no_of_nodes"])] 
            n_V = case_in["no_of_nodes"]#node
            n_E = case_in["no_of_edges"]
        else:
            raise Exception("input is wrong. please check.")

        for i_v in range(n_V):
            target = case_in["node"][i_v,:]
            set_V[i_v]["idx_v"] = target[0]
            set_V[i_v]["idx_t"] = target[1]
            #find children and father
            set_V[i_v]["set_child"] = [case_in["edge_to"][e_ind,0] for e_ind in range(n_E) if np.array_equal(case_in["edge_from"][e_ind], target)]
            set_V[i_v]["set_father"] = [case_in["edge_from"][e_ind,0] for e_ind in range(n_E) if np.array_equal(case_in["edge_to"][e_ind], target)]
            
        return set_V
