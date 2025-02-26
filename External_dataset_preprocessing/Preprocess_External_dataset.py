"""
File: Preprocess_External_Dataset.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de
Gitlab: https://gitlab.com/qumphy
Description: ..
SPDX-License-Identifier: EUPL-1.2
"""

import pandas as pd
from pathlib import Path
import numpy as np
import scipy.io as sio




#insert absolute paths to output and input folders
directory_output=Path("insert/your/path/output")

directory_bcg=Path("insert/your/path/bcg_dataset")
directory_uci=Path("insert/your/path/uci2_dataset")
directory_sensors=Path("insert/your/path/sensors_dataset")
directory_ppgbp=Path("insert/your/path/ppgbp_dataset")





def process_folder(path, output_path):
    path=Path(path)
    output_path=Path(output_path)
    df_feat=[]
    df_signal=[]
    df_signal_mabp=[]
    lst_signal=[]
    lst_signal_mabp=[]
    fs=sorted(list(path.glob('*.csv')))
    
    
    if path==directory_ppgbp:
        for f in fs:
            #features
            print(f)
            df_feat.append(pd.read_csv(f))
            df_feat[-1]["fold"]=int(f.stem.split("_")[-1])
                #signals
                #for x in ["signal","signal_mabp"]:
            for x in ["signal"]:    
                tmp={}
                f_signal=f.parent/(f.stem.replace("feat",x)+".mat")
                test = sio.loadmat(f_signal)
                for k in list(test.keys())[3:]:
                    if(not k in ["signal","abp_signal"]):
                        tmp[k.lower()]=list(test[k])
                    else:
                        fname=output_path/(f_signal.stem+"_"+k+".npy")
                        np.save(fname,test[k])
                        if(x=="signal"):
                            lst_signal.append(fname)
                        else:
                            lst_signal_mabp.append(fname)
                tmp=pd.DataFrame(tmp)
                if(x=="signal"):
                    df_signal.append(tmp)
                else:
                    df_signal_mabp.append(tmp)
    
        df_feat= pd.concat(df_feat)
        df_signal= pd.concat(df_signal)
      #  df_signal_mabp = pd.concat(df_signal_mabp)
        for k in df_signal.columns:
            df_signal[k]=df_signal[k].apply(lambda x: x[0])
        #for k in df_signal_mabp.columns:
         #   if(k!="class_limits"):
          #      df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: x[0])
        df_feat.columns=[k.lower() for k in df_feat.columns]
    
        for k in ["trial","patient"]:
            #df_feat[k]=df_feat[k].apply(lambda x: x.strip())
            #df_signal[k]=df_signal[k].apply(lambda x: x.strip())
            #df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: x.strip())
            df_feat[k]=df_feat[k].apply(lambda x: str(x).strip())
            df_signal[k]=df_signal[k].apply(lambda x: str(x).strip())
        #    df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: str(x).strip())
    
        #full metadata
       # df_all=df_signal.join(df_signal_mabp.drop(["patient","sp","dp"],axis=1).set_index("trial"),on="trial")
        #df_all=df_all.join(df_feat.drop(["patient","sp","dp"],axis=1).set_index("trial"),on="trial")
        df_all=df_signal.join(df_feat.drop(["patient","sp","dp"],axis=1).set_index("trial"),on="trial")
        
        df_all["dataset"]=path.stem


    else:

                for f in fs:
                    #features
                    print(f)
                    df_feat.append(pd.read_csv(f))
                    df_feat[-1]["fold"]=int(f.stem.split("_")[-1])
                        #signals
                    for x in ["signal","signal_mabp"]:
                    
                        tmp={}
                        f_signal=f.parent/(f.stem.replace("feat",x)+".mat")
                        test = sio.loadmat(f_signal)
                        for k in list(test.keys())[3:]:
                            if(not k in ["signal","abp_signal"]):
                                tmp[k.lower()]=list(test[k])
                            else:
                                fname=output_path/(f_signal.stem+"_"+k+".npy")
                                np.save(fname,test[k])
                                if(x=="signal"):
                                    lst_signal.append(fname)
                                else:
                                    lst_signal_mabp.append(fname)
                        tmp=pd.DataFrame(tmp)
                        if(x=="signal"):
                            df_signal.append(tmp)
                        else:
                            df_signal_mabp.append(tmp)
    
                df_feat= pd.concat(df_feat)
                df_signal= pd.concat(df_signal)
                df_signal_mabp = pd.concat(df_signal_mabp)
                for k in df_signal.columns:
                    df_signal[k]=df_signal[k].apply(lambda x: x[0])
                for k in df_signal_mabp.columns:
                    if(k!="class_limits"):
                        df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: x[0])
                df_feat.columns=[k.lower() for k in df_feat.columns]
            
                for k in ["trial","patient"]:
                    #df_feat[k]=df_feat[k].apply(lambda x: x.strip())
                    #df_signal[k]=df_signal[k].apply(lambda x: x.strip())
                    #df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: x.strip())
                    df_feat[k]=df_feat[k].apply(lambda x: str(x).strip())
                    df_signal[k]=df_signal[k].apply(lambda x: str(x).strip())
                    df_signal_mabp[k]=df_signal_mabp[k].apply(lambda x: str(x).strip())
            
                #full metadata
                df_all=df_signal.join(df_signal_mabp.drop(["patient","sp","dp"],axis=1).set_index("trial"),on="trial")
                df_all=df_all.join(df_feat.drop(["patient","sp","dp"],axis=1).set_index("trial"),on="trial")
                
                df_all["dataset"]=path.stem    
    
    return df_all, lst_signal, lst_signal_mabp






df_bcg,files_bcg,_=process_folder(directory_bcg,"put/your/path")
df_uci,files_uci,_=process_folder(directory_uci,"put/your/path")
df_sensors,files_sensors,_=process_folder(directory_sensors,"put/your/path")
df_ppgbp,files_ppgbp,_=process_folder(directory_ppgbp,"put/your/path")



convert signals to memmep

from timeseries_utils import *

files_all = files_bcg+files_uci+files_sensors+files_ppgbp
files_all_ppg=[x for x in files_all if not x.stem.endswith("abp_signal")]#remove abps
print(files_all_ppg)

npys_to_memmap_batched(
    files_all_ppg,
    directory_output/"memmap.npy",
    max_len=0,
    delete_npys=True,
    batched_npy=True,
)







