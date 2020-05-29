import pandas as pd
import torch
import os
from covid_19_test import save_torch,cat_fix,normalize
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
binary = [
            'rf_pdiab',
          'rf_gdiab',
          'rf_phype',
          'rf_ghype',
          'rf_ehype',
          'rf_ppterm',
          'rf_inftr',
          'rf_cesar',
          'rf_cesarn',
          'ip_gon',
          'ip_syph',
          'ip_chlam',
          'ip_hepatb',
          'ip_hepatc',
          'ob_ecvs',
            'ld_indl',
          'ld_augm',
        'ld_ster',
          'ld_antb',
          'ld_chor',
          'ld_anes',
        'dmeth_rec',
        'mm_mtr',
          'mm_plac',
          'mm_rupt',
          'mm_uhyst',
          'mm_aicu',
            'mtran',
          'sex',
          'ab_aven1',
          'ab_aven6',
          'ab_nicu',
          'ab_surf',
          'ab_anti',
          'ab_seiz',
        'ca_anen',
          'ca_mnsb',
          'ca_cchd',
          'ca_cdh',
          'ca_omph',
          'ca_gast',
          'ca_limb',
          'ca_cleft',
          'ca_clpal',
          'ca_downs',
          'ca_disor',
          'ca_hypo',
            'itran',
            'bfed',
          ]
categorical = ['dob_mm',
               'dob_wk',
               'bfacil3',
               'mager9',
               'mbstate_rec',
               'restatus',
               'mrace6',
               'meduc',
               'frace6',
               'feduc',
               'illb_r11',
               'ilp_r11',
               'precare5',
               'wic',
                'bmi_r',
               'wtgain_rec',
               'me_pres',
               'me_rout',
               'me_trial',
                'attend',
               'pay',
               'dplural',
               'dlmp_mm',
               'rf_fedrg',
               'rf_artec',

               ]
continuous = ['priorlive',
              'priordead',
              'priorterm',
              'previs',
'cig_0', 'cig_1', 'cig_2', 'cig_3',
'bmi','pwgt_r','dbwt','wtgain','rf_cesarn','apgar5','combgest'
              ]

bin_cat = binary + categorical

def transform(df):
    col_names = df.columns.tolist()
    _cat = []
    for el in col_names:
        if el in bin_cat:
            _cat.append(el)
    if _cat:
        cat_data = cat_fix(df[_cat])
        df = df.drop(columns=_cat)
        df = pd.concat([df,cat_data],axis=1)
    return normalize(df)
if __name__ == '__main__':
    seeds = 100
    for s in range(seeds):
        if not os.path.exists('filtered_infant_data.csv'):
            df = pd.read_csv("natl2017.csv")
            print(df.columns)
            list_of_var = [
                'dob_yy',
                'DOB_TT',
                'BFACIL',
                'F_FACILITY',
                'MAGE_IMPFLG',
                'MAGE_REPFLG',
                'MAGER',
                'MAGER14',
                'MRACE31',
                'MRACE15',
                'MBRACE',
                'MRACEIMP',
                'MHISP_R',
                'F_MHISP',
                'MRACEHISP',
                'MAR_IMP',
                'F_MAR_P',
                'F_MEDUC',
                'FAGERPT_FLG',
                'FAGECOMB',
                'FRACE31',
                'FRACE15',
                'FHISP_R',
                'F_FHISP',
                'FRACEHISP',
                'F_FEDUC',
                'ILOP_R',
                'ILP_R',
                'F_MPCB',
                'PREVIS_REC',
                'F_TPCV',
                'F_WIC',
                'CIG0_R',
                'CIG1_R',
                'CIG2_R',
                'CIG3_R',
                'F_CIGS_0',
                'F_CIGS_1',
                'F_CIGS_2',
                'F_CIGS_3',
                'CIG_REC',
                'F_TOBACO',
                'F_M_HT',
                'F_PWGT',
                'F_DWGT',
                'F_WTGAIN',
                'F_RF_PDIAB',
                'F_RF_GDIAB',
                'F_RF_PHYPER',
                'F_RF_GHYPER',
                'F_RF_ECLAMP',
                'F_RF_PPB',
                'f_RF_INFT',
                'F_RF_INF_DRG',
                'F_RF_INF_ART',
                'F_RF_CESAR',
                'F_RF_NCESAR',
                'F_IP_GONOR',
                'F_IP_SYPH',
                'F_IP_CHLAM',
                'F_IP_HEPATB',
                'F_IP_HEPATC',
                'F_OB_SUCC',
                'F_OB_FAIL',
                'F_LD_INDL',
                'F_LD_AUGM',
                'F_LD_STER',
                'F_LD_ANTB',
                'F_LD_CHOR',
                'F_LD_ANES',
                'F_ME_PRES',
                'F_ME_ROUT',
                'F_ME_TRIAL',
                'F_DMETH_REC',
                'F_MM_MTR',
                'F_MM_PLAC',
                'F_MM_RUPT',
                'F_MM_UHYST',
                'F_MM_AICU',
                'FILLER_MM',
                'F_PAY',
                'F_PAY_REC',
                'APGAR5R',
                'F_APGAR5',
                'APGAR10R',
                'IMP_PLUR',
                'IMP_SEX',
                'LMPUSED',
                'BWTR12',
                'BWTR4',
                'F_AB_VENT',
                'F_AB_VENT6',
                'F_AB_NIUC',
                'F_AB_SURFAC',
                'F_AB_ANTIBIO',
                'F_AB_SEIZ',
                'F_CA_ANEN',
                'F_CA_MENIN',
                'F_CA_HEART',
                'F_CA_HERNIA',
                'F_CA_OMPHA',
                'F_CA_GASTRO',
                'F_CA_LIMB',
                'F_CA_CLEFTLP',
                'F_CA_CLEFT',
                'F_CA_DOWNS',
                'F_CA_CHROM',
                'F_CA_HYPOS',
                'F_BFED',
                'compgst_imp',
                'obgest_flg',
                'mar_p'
            ]
            for el in list_of_var:
                try:
                    df = df.drop([el.lower()],axis=1)
                except Exception as e:
                    print(e)
            df.to_csv('filtered_infant_data.csv')
        else:
            if not os.path.exists(f'filtered_infant_data_stratified_subsampled_seed={s}.csv'):
                df = pd.read_csv('filtered_infant_data.csv',index_col=0)
                df = df.dropna(axis=1)
                df = df[df['ilive'].isin(['Y','N'])]
                for el in continuous:
                    df = df[~df[el].isin([99,999.9,999,9999])]
                for el in binary:
                    df = df[~df[el].isin(['U',9])]
                print(df.shape)
                df_dead = df[df['ilive']=='N'].sample(n=5000, random_state=s)
                df_alive = df[df['ilive']=='Y'].sample(n=5000, random_state=s)
                df_subsampled = pd.concat([df_alive,df_dead],axis=0)
                df_subsampled.to_csv(f'filtered_infant_data_stratified_subsampled_seed={s}.csv')
            else:
                df = pd.read_csv(f'filtered_infant_data_stratified_subsampled_seed={s}.csv',index_col=0)
                df['infant_died_at_birth'] = df['ilive'].apply(lambda x: 1 if x=='N' else 0)

                x = ['meduc']
                y = ['infant_died_at_birth']
                z = [
                    'mager9',
                    'mrace6',
                    'mbstate_rec',
                    'restatus','cig_0', 'cig_1', 'cig_2', 'cig_3',
    'bmi','pwgt_r','dbwt','wtgain','rf_cesarn','apgar5','combgest'
                    ]
                X_pd = df[x]
                Y_pd = df[y]
                Z_pd = df[z]
                X_pd = transform(X_pd)
                Y_pd = transform(Y_pd)
                Z_pd = transform(Z_pd)
                save_torch(X_pd, Y_pd, Z_pd, './infant_mortality_1/', f'data_seed={s}.pt')
