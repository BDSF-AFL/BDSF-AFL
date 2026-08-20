import pandas as pd
import numpy as np

jnt = pd.read_csv('logs/phase3_verify/joint_COMPOUND_42_updates.csv')
leg = pd.read_csv('logs/phase3_verify/legacy_COMPOUND_42_updates.csv')

byz_ids = {0, 1, 2, 3}
jnt['is_byz'] = jnt['client_id'].isin(byz_ids)
leg['is_byz'] = leg['client_id'].isin(byz_ids)

print('=== 1. CONTINUOUS SIGNAL DISTRIBUTIONS (JOINT MODE) ===')
for col in ['sim_global', 'sim_self_mean', 'sim_anchor', 'norm_raw', 'g_i', 'fence_margin']:
    h_vals = jnt[~jnt['is_byz']][col].dropna()
    b_vals = jnt[jnt['is_byz']][col].dropna()
    print(f'\nFeature: {col}')
    print(f'  Honest:    mean={h_vals.mean():.4f}, std={h_vals.std():.4f}, min={h_vals.min():.4f}, median={h_vals.median():.4f}, max={h_vals.max():.4f}, count={len(h_vals)}')
    print(f'  Byzantine: mean={b_vals.mean():.4f}, std={b_vals.std():.4f}, min={b_vals.min():.4f}, median={b_vals.median():.4f}, max={b_vals.max():.4f}, count={len(b_vals)}')

print('\n=== 2. QUARANTINE RESOLUTION TIMELINE (JOINT MODE) ===')
quar_entries = jnt[jnt['status'] == 'QUARANTINE']
rel_entries = jnt[jnt['reason'] == 'QUARANTINE_RELEASE_ACCEPT']
print(f'Total Quarantined events: {len(quar_entries)}')
print(f'Total Quarantine Released events: {len(rel_entries)}')
print(f'Quarantine Released by Client ID: {rel_entries["client_id"].value_counts().to_dict()}')

print('\n=== 3. REPUTATION DYNAMICS ===')
leg_h = leg[~leg['is_byz']].groupby('client_id').last()
jnt_h = jnt[~jnt['is_byz']].groupby('client_id').last()
leg_b = leg[leg['is_byz']].groupby('client_id').last()
jnt_b = jnt[jnt['is_byz']].groupby('client_id').last()

print(f'  Legacy Honest:    Integrity mean={leg_h["I_i"].mean():.4f}, Pace mean={leg_h["P_i"].mean():.4f}')
print(f'  Joint Honest:     Integrity mean={jnt_h["I_i"].mean():.4f}, Pace mean={jnt_h["P_i"].mean():.4f}')
print(f'  Legacy Byzantine: Integrity mean={leg_b["I_i"].mean():.4f}, Pace mean={leg_b["P_i"].mean():.4f}')
print(f'  Joint Byzantine:  Integrity mean={jnt_b["I_i"].mean():.4f}, Pace mean={jnt_b["P_i"].mean():.4f}')
