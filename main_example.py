import numpy as np
from scenredpy import Cls_scenred, Cls_makeset

sr_instance = Cls_scenred()

sr_instance.import_data("dataset/sample_50.h5")
sr_instance.prepare_data()

#sr_instance.scenario_reduction(dist_type="cityblock", fix_node=1, tol_node=np.linspace(1,24, 24)) fix_node_no

sr_instance.scenario_reduction(dist_type="cityblock",fix_prob=1, tol_prob=np.linspace(0, 0.2, 24)) #fix_prob_tol

sr_instance.draw_reduced_scenario()

sr_instance.sort_result()

#set_V = Cls_makeset.generate_set(sr_instance.get_instance())
