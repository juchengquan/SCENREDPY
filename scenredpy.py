# -*- coding: utf-8 -*-
"""
Last modified on 20190311
@author: cqju 
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import copy 
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale
from copy import deepcopy

class Cls_scenred():
    def __init__(self, **kwargs):
        pass
    
    def show_attributes(self):
        print(self.__dict__.keys())
    
    def import_data(self, filename, **kwargs):
        if isinstance(filename, str) == False:
            raise Exception("No such file in the directory.")
        
        f = h5py.File(filename, "r")
        if "data" not in f.keys():
            raise Exception("No 'data' in input file. check again.")
        else:
            self.name_list = list(f["data"].keys())
        
        d = dict()
        for name in self.name_list:
            d[name] = f["data"][name][()] 
            #[()] is to  transform data into ndarray. 
            #see https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
            
        self.data = d

    def prepare_data(self, **kwargs):
        """
        prepare information of data
        """
        if isinstance(self.data, dict) == False:  #input data should be a dict.
            raise Exception("The input data is not a dictionary.")
        
        self.dim_DAT = len(self.data)
        
        self.data_agg =  np.hstack(self.data.values()) 
        self.dim_M, self.dim_N = self.data_agg.shape[0], int(self.data_agg.shape[1] /self.dim_DAT)


    def scenario_reduction(self, **kwargs):    
        #prepare data:
        self.prepare_data()

        dist_type = kwargs.get("dist_type", "cityblock")
        verbose_mode = kwargs.get("verbose", 0)
        #selection:
        sel = dict()
        sel["fix_prob"] = kwargs.get("fix_prob", 0)
        sel["fix_node"] = kwargs.get("fix_node", 0)
        if np.sum(ele_val for ele_val in sel.values() ) == 0:
            print(1)
            raise Exception("At least one constraint should be set.")
        else:
            tol_prob = kwargs.get("tol_prob", np.linspace(0, 0, self.dim_N) ) #by default, no one should be reduced.
            tol_node = kwargs.get("tol_node", np.full(self.dim_N, 1) ) #by default, no of nodes must be larger than 1.

        dim_M, dim_N = self.dim_M, self.dim_N

        data_agg = self.data_agg

        #PRE-DEFINED
        nodes_left = data_agg.shape[0]

        #prob matrix
        m_epo_prob = np.full([dim_M, dim_N], 1)
        
        #distance matrix
        #get distance
        data_normalized = scale(data_agg, axis=0) #z-score 
        
        m_dist = cdist(data_normalized, data_normalized, dist_type, p=2) # distance #'cityblock' 'minkowski', p=2 
        m_dist_aug = m_dist+ np.eye(np.size(data_agg, 0))*(1+np.max(m_dist))
        
        #distance probability matrixs
        m_dist_prob = np.full(m_dist_aug.shape,True,dtype=bool)
        
        # residue matrix 
        m_epo_res = np.full([dim_M, dim_N], True, dtype=bool)
        
        # linkage matrix in nodes:
        m_link = np.full(m_dist_aug.shape, 0) 
        # linkage matrix in epoch:
        m_epo_link = -np.full([dim_M, dim_N], 1) 
        
        data_out = copy.deepcopy( self.data )
        
        ## START ITERATION:
        if dim_M ==1:
            raise Exception("there is only one instance.")
        
        for epo in np.flip(np.arange(dim_N), axis=0): #backward
            dp_limit = 1 * tol_prob[epo] * np.min([i for i in np.sum(m_dist * m_dist_prob, axis=0) if i >0])
            dp = 0
            
            counter = True
            while counter:
                m_dist_aug[ ~m_epo_res[:,epo], : ] = np.Inf
                m_dist_aug[ :, ~m_epo_res[:,epo] ] = np.Inf
                
                c_min_M = np.min(m_dist_aug, axis=1)
                c_min_M_idx = np.argmin(m_dist_aug, axis=1)
                
                #min z
                arr_z = c_min_M * m_epo_prob[:,epo] 
                for i in np.arange(arr_z.shape[0]):
                    if np.isnan(arr_z[i]) == True or arr_z[i] ==0 :
                        arr_z[i] = np.Inf
                
                del_val = np.min(c_min_M)
                del_idx = np.argmin(c_min_M)
                
                aug_idx = c_min_M_idx[del_idx]
                
                dp += del_val 
                
                # use mode selection:
                if dp <= dp_limit and nodes_left > tol_node[epo] :
                    #transfer probability:
                    m_epo_prob[aug_idx, epo] += m_epo_prob[del_idx, epo]
                    
                    #delete probability with related index:
                    m_dist_prob[del_idx, :] = False
                    m_dist_prob[:, del_idx] = False
                    
                    #delete probability in epoch:
                    m_epo_res[del_idx, epo]= False
                    m_epo_prob[del_idx, epo]= 0
                    
                    #calculate nodes left
                    nodes_left = np.sum(m_epo_prob[:,epo]>0)
                    
                    #recunstruct link matrix in nodes:
                    m_link[aug_idx, del_idx] = 1
                    #inherit deleted index
                    m_link[aug_idx, m_link[del_idx,:]==1] = 1 
                    
                    #recunstruct link matrix in epoch:
                    m_epo_link[del_idx, epo] = aug_idx
                    #check if this del_idx is a formal aug_idx:
                    m_epo_link[np.where(m_epo_link[:, epo] == del_idx ), epo] = aug_idx

                    to_merge_idx = np.where(m_link[del_idx,:])
                    #remove data from deleted scenario:
                    for ele_data in data_out.values():
                        ele_data[del_idx, 0:epo+1] = ele_data[aug_idx, 0:epo+1] 
                    #inherit from deleted index, if it has information from other samples:
                        for ele in to_merge_idx:
                            ele_data[ele, 0:epo+1] = ele_data[aug_idx, 0:epo+1] 
                else: 
                    #close current iteration
                    counter = False
                    dp -= del_val
            #end while
            
            if epo >0:
            #Update available scenarios in the previous epoch:
                m_epo_res[:,epo-1] = m_epo_res[:,epo]
                m_epo_prob[:,epo-1] = m_epo_prob[:,epo]
            if verbose_mode == 1 :
                print("epo={1}: {0} nodes left ".format(nodes_left, epo))
        
        ## END ITERATION
        self.data_out = data_out
        self.m_epo_res = m_epo_res
        self.m_epo_link = m_epo_link
        self.m_epo_prob = m_epo_prob

    def sort_result(self):
        dim_M, dim_N = self.dim_M, self.dim_N
        
        x_epoch = [dict() for epo in range(dim_N)]
        p_epoch = [dict() for epo in range(dim_N)]
        loc_epoch = [dict() for epo in range(dim_N)] 
        
        for epo in range(dim_N):
            p_epoch[epo] = self.m_epo_prob[ self.m_epo_prob[:,epo] != 0, epo ] 
            loc_epoch[epo] = np.where( self.m_epo_prob[:,epo] != 0 )[0]
            for key in self.name_list:
                x_epoch[epo][key] = self.data_out[key][ self.m_epo_prob[:,epo] != 0, epo ]
                
            for ele in loc_epoch[epo]:
                self.m_epo_link[ele,epo] = ele

        op_prob = np.concatenate(p_epoch) / dim_M
        op_data = {key: np.concatenate([d.get(key) for d in x_epoch]) for key in self.name_list}

        #np.concatenate(x_epoch)

        #find edges and tol_node with old index: 
        ip_node = np.array([])
        for epo in range(dim_N):
            for ele in loc_epoch[epo]:
                ip_node = np.concatenate( (ip_node, [ele, epo]) )
        ip_node = np.reshape(ip_node, (-1,2)).astype(int)

        ip_edge = np.array([])
        for epo in range(dim_N -1):
            for i in range(dim_M):
                if self.m_epo_link[i,epo] != -1:
                    temp_p = [self.m_epo_link[i,epo], epo, i, epo+1] #from, to 
                    ip_edge = np.concatenate( (ip_edge, temp_p) )
        ip_edge = np.reshape(ip_edge, (-1,2)).astype(int)
        
        # find edges and tol_node with new index:
        op_node = copy.deepcopy(ip_node)
        op_edge = copy.deepcopy(ip_edge) 
        
        op_node[:,0] = np.arange( sum(np.shape(loc_epoch[epo])[0] for epo in range(dim_N)) ) # no. of nodes remained
        for i_node in range(ip_node.shape[0]): 
            for i_edge in range(ip_edge.shape[0]):
                if np.array_equal(ip_node[i_node], ip_edge[i_edge]):
                    op_edge[i_edge] = op_node[i_node]

        op_edge = np.reshape(op_edge, (-1,4)).astype(int)
        op_edge_from = op_edge[:,[0,1]]
        op_edge_to = op_edge[:,[2,3]]
        
        #self as output:
        self.op_prob = op_prob
        self.op_data = op_data
        
        self.op_node = op_node
        self.op_edge = op_edge
        self.op_edge_from = op_edge_from
        self.op_edge_to = op_edge_to
        
        self.op_horizon = np.reshape(np.arange(dim_N), (-1, 1))

    #draw the scenadio reduction figure
    def draw_reduced_scenario(self):
        #mainFig = plt.figure(figsize=(8,8))
        for key in self.name_list:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            d_draw = self.data_out[key][self.m_epo_res[:,-1],:].transpose()
            ax.plot(np.linspace(1,self.dim_N, self.dim_N), d_draw, color='gray', linewidth=0.5, alpha=0.5)
            ax.set_xlim((1,self.dim_N))
            ax.set_title(key)

### external functions
    def get_instance(self):
        new_ins = dict()
        new_ins["node"] = self.op_node
        new_ins["no_of_time"] = self.dim_N
        new_ins["no_of_nodes"] = self.op_node.shape[0] + 1
        
        new_ins["edge"] = self.op_edge
        new_ins["edge_from"] = self.op_edge_from
        new_ins["edge_to"] = self.op_edge_to
        new_ins["prob"] = self.op_prob
        new_ins["data"] = self.op_data
        #return value
        return new_ins
    
    def get_data_instance(self):
        new_ins = dict()
        new_ins["prob"] = deepcopy(self.op_prob)
        new_ins["data"] = deepcopy(self.op_data)
        
        #insert root node:
        new_ins["prob"] = np.insert(new_ins["prob"], 0, 1)
        for ele in new_ins["data"]:
            new_ins["data"][ele] = np.insert(new_ins["data"][ele], 0, 0)
        #return value
        return new_ins


    @staticmethod
    def read_data(filename, **kwargs):
        if isinstance(filename, str) == False:
            raise Exception("No such file in the directory.")
        
        f = h5py.File(filename, "r")
        if "data" not in f.keys():
            raise Exception("No 'data' in input file. check again.")
        else:
            name_list = list(f["data"].keys())
        
        d = dict()
        for name in name_list:
            d[name] = f["data"][name][()] 
            #[()] is to  transform data into ndarray. 
            #see https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
            
        return d

class Cls_makeset():
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def generate_set(case_in):
        if "node" in case_in.keys() and "edge" in case_in.keys():
            n_V = len(case_in["node"])
            n_E = len(case_in["edge"])
            set_V = [dict() for i in range(n_V)] 
        else:
            raise Exception("input is wrong. please check.")
        
        set0_child = []
        for i_v in range(n_V):
            target = case_in["node"][i_v,:]
            set_V[i_v]["idx_v"] = target[0] + 1 # modify node index
            set_V[i_v]["idx_t"] = target[1] + 1 # modify time index
            #find children and father
            set_V[i_v]["set_child"] = [case_in["edge_to"][e_ind,0] +1 for e_ind in range(n_E) if np.array_equal(case_in["edge_from"][e_ind], target)]
            set_V[i_v]["set_father"] = [case_in["edge_from"][e_ind,0] +1 for e_ind in range(n_E) if np.array_equal(case_in["edge_to"][e_ind], target)]
            if set_V[i_v]["set_father"]:
                pass
            else:
                set_V[i_v]["set_father"] = [0]
                set0_child.append(set_V[i_v]["idx_v"])
        # add COMPLEMENTARY NODE AT BEGINNING
        set0={"idx_v":0, "idx_t":0, "set_child": set0_child, "set_father": []}
        set_V.insert(0, set0)
        #return value
        return set_V

    
    

