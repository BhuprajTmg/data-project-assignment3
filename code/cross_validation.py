"""Supplementary 5-fold validation; run: python code/cross_validation.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, cross_validate
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'cleaned'
OUT = ROOT / 'validation_results'
OUT.mkdir(exist_ok=True)
SPECS = [
 ('Model 2.1','model21_goal_diff.csv','goal_diff',['appearances_diff','days_into_tournament','youth_share_diff','age_spread_diff'],False),
 ('Model 2.2','model22_team_goals.csv','goals',['team_appearances','days_into_tournament','opp_appearances','is_uefa'],True),
]
rows=[]
for name,filename,target,features,grouped in SPECS:
 df=pd.read_csv(DATA/filename)
 cv=GroupKFold(n_splits=5) if grouped else KFold(n_splits=5,shuffle=True,random_state=42)
 scores=cross_validate(LinearRegression(),df[features],df[target],cv=cv,groups=df['match_id'] if grouped else None,scoring={'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error'})
 for i in range(5):
  rows.append({'model':name,'fold':i+1,'r2':scores['test_r2'][i],'mae':-scores['test_mae'][i],'rmse':-scores['test_rmse'][i],'grouped_by_match':grouped})
 print(name, 'rows',len(df),'grouped',grouped)
 for metric in ['r2','mae','rmse']:
  a=np.array([r[metric] for r in rows if r['model']==name]); print(metric, 'mean',round(a.mean(),4),'std',round(a.std(ddof=1),4),'folds',a.round(3))
pd.DataFrame(rows).to_csv(OUT/'five_fold_cv.csv',index=False)
