import easypyxl
import os
import numpy as np
import matplotlib.pyplot as plt

cfg_idx=[51,92,96,101,102]
methods=['DI-MI-TI','RDI-MI-TI','RDI-MI-TI-VT','ODI-MI-TI','ODI-MI-TI-VT']
styles=['r-','b']
result_dir="results/"
target_models=['VGG-16','RN-18', 'RN-50', 'DN-121', 'Inc-v3', 'Inc-v4', 'Mob-v2','IR-v2',  'Adv-Inc-v3', 'Ens-adv-IR-v2']
source_model_names=['RN-50','VGG-16', 'DN-121', 'Inc-v3']

display_idx=[0,1,2,3,4,5,6,7,8,9]

num_iterations=300
values=np.zeros((len(cfg_idx),len(source_model_names),len(target_models),num_iterations//20+1), dtype=np.float32)






workbook=[]
cursor=[[] for i in range(num_iterations//20)]

for i in range(len(cfg_idx)):
    workbook.append(easypyxl.Workbook(result_dir+"NEW_EXP_"+str(cfg_idx[i])+".xlsx", backup=False))
    for j in range(num_iterations//20):
        cursor[i].append(workbook[i].new_cursor("Succ_"+str((j+1)*20), "C2", len(target_models), reader=True))
        title=cursor[i][j].read_line()
        if i==0 and j==0:
            for j in display_idx:
                print(title[j],end=', ')


for i in range(len(source_model_names)):
    print(source_model_names[i])
    for c in range(len(cfg_idx)):
        for j in range(num_iterations//20):
            for k in range(len(target_models)):
                item=cursor[c][j].read_cell()
                values[c][i][k][j+1]=item



for s in range(4):
    for t in range(10):
        max=0
        target_idx=t
        source_idx=s
        # target_idx=6
        # source_idx=1
        # target_idx=2
        # source_idx=1
        colors = ['tab:purple',
                    'tab:blue',
                    'tab:green',
                    'tab:orange',
                    'tab:red',]
        markers = ['s',
            '^',
                    '^',
                    'o',
                    'o',]
        # colors = ['#1f77b4',
        #             '#ff7f0e',
        #             '#2ca02c',
        #             '#d62728',]
        x=np.arange(0,num_iterations+20,20)
        # summarize history for loss
        plt.figure(figsize=(4, 3), dpi=300)
        plt.tight_layout()

        for i in range(len(cfg_idx)):
            plt.plot(x,values[i,source_idx,target_idx],color=colors[i],marker=markers[i],linewidth=2,aa=True,markersize=4)
            if max<values[i,source_idx,target_idx].max():
                max=values[i,source_idx,target_idx].max()
        #plt.rcParams.update({'font.size': 15})
        plt.title(source_model_names[source_idx]+' (Source) → '+target_models[target_idx] + ' (Target)')
        plt.ylabel('Attack success rate (%)', fontsize=13)
        plt.xlabel('Iteration', fontsize=12)
        #plt.grid(True)
        plt.grid(color='gainsboro', linestyle='-', linewidth=1)
        plt.xlim(0, 300)

        plt.ylim(0, (max//10+1)*10)
        # plt.ylim(0)
        plt.legend(methods, loc='best')#, ncol=1,frameon=False
        plt.tight_layout(pad=1.0)
        #plt.show()
        
        plt.savefig('Supp/'+str(s)+'_'+str(t)+'.pdf')