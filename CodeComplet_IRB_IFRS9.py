print("="*80)

print("Auteur: Rodrigue KOFFI")

print("PROJET RISQUE DE CRÉDIT - IRB-A / IFRS9 / CRR2/CRR3")

print("="*80)

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import os
print(os.getcwd())  # get dosiier de travail


from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

def force_numeric(series):
    """
    Convertion auto
    """
    if series is None:
        return pd.Series([], dtype=float)
    
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    if hasattr(series, 'cat') or (hasattr(series, 'dtype') and series.dtype.name == 'category'):
        try:
            return series.cat.codes.astype(float)
        except:
            return series.astype(float)
    
    if pd.api.types.is_object_dtype(series):
        try:
            return pd.to_numeric(series, errors='coerce').fillna(0)
        except:
            return series.astype('category').cat.codes.astype(float)
    
    try:
        return series.astype(float)
    except:
        return series

def force_numeric_df(df, columns=None):
    """Applique force_numeric  au moment opportun"""
    df_result = df.copy()
    if columns is None:
        columns = df_result.columns
    for col in columns:
        if col in df_result.columns:
            df_result[col] = force_numeric(df_result[col])
    return df_result

print("="*80)
print("PROJET RISQUE DE CRÉDIT - IRB-A / IFRS9 / CRR2/CRR3")

print("="*80)

# ============================================================
# PARTIE 1 : CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================
print("\n" + "="*80)
print("PARTIE 1 : CHARGEMENT ET NETTOYAGE DES DONNÉES")
print("="*80)

print("\n1.1 CHARGEMENT DU FICHIER EXCEL...")

data_path = Path("data")
file_path = data_path / 'german_credit_data.xlsx'
df_raw = pd.read_excel(file_path, sheet_name='german_credit_data(1)')
print(f"Data OK : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")

df_raw.columns = ['age', 'sex', 'job', 'housing', 'savingAccounts', 
                  'checkingAccount', 'creditAmount', 'duration', 'purpose', 'risk']

print("\n1.2 NETTOYAGE ET PRÉPARATION...")

def cleanData(df):
    df_clean = df.copy()
    df_clean['defaultFlag'] = (df_clean['risk'] == 'bad').astype(int)
    df_clean['sexNum'] = (df_clean['sex'] == 'male').astype(int)
    
    housingMap = {'own': 1, 'rent': 2, 'free': 3}
    df_clean['housingNum'] = df_clean['housing'].map(housingMap)
    
    savingMap = {'NA': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
    df_clean['savingAccountsNum'] = df_clean['savingAccounts'].map(savingMap).fillna(0)
    
    checkingMap = {'NA': 0, 'little': 1, 'moderate': 2, 'rich': 3}
    df_clean['checkingAccountNum'] = df_clean['checkingAccount'].map(checkingMap).fillna(0)
    
    df_clean['logCreditAmount'] = np.log1p(df_clean['creditAmount'])
    df_clean['loanPerYear'] = force_numeric(df_clean['creditAmount']) / force_numeric(df_clean['duration'])
    df_clean['ageGroup'] = pd.cut(df_clean['age'], bins=[0, 25, 35, 50, 100], 
                                   labels=['young', 'adult', 'middle', 'senior'])
    
    print(f"  Default rate : {df_clean['defaultFlag'].mean():.2%}")
    return df_clean

df_clean = cleanData(df_raw)




# ============================================================
# PARTIE 2 : FEATURE ENGINEERING & DONNÉES MACRO
# ============================================================
print("\n" + "="*80)
print("PARTIE 2 : FEATURE ENGINEERING & DONNÉES MACROÉCONOMIQUES")
print("="*80)

print("\n2.1 GÉNÉRATION DES DONNÉES MACROÉCONOMIQUES...")

years = np.arange(2000, 2020)
macroData = pd.DataFrame({
    'year': years,
    'unemploymentRate': [7.8, 7.6, 8.6, 9.3, 9.8, 10.6, 9.8, 8.7, 7.5, 7.8,
                         7.1, 6.0, 5.5, 5.3, 5.0, 4.6, 4.1, 3.8, 3.4, 3.1],
    'gdpGrowth': [2.9, 2.0, 1.5, 1.0, 0.5, 1.5, 3.0, 2.5, 1.0, -5.0,
                  3.0, 2.5, 1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0],
    'interestRate': [3.5, 3.3, 3.0, 2.8, 2.5, 2.7, 3.1, 3.8, 4.0, 3.2,
                     2.0, 1.5, 0.8, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    'hpiIndex': [100, 101, 102, 103, 104, 105, 108, 112, 115, 118,
                 120, 125, 130, 135, 140, 148, 155, 162, 170, 178]
})

np.random.seed(42)
df_clean['year'] = np.random.choice(years, len(df_clean))
df_clean = df_clean.merge(macroData, on='year', how='left')
df_clean['isOutOfTime'] = df_clean['year'] >= 2017
print(f"  Train time: {df_clean[~df_clean['isOutOfTime']].shape[0]} lignes")
print(f"  Time Out-of-Time : {df_clean[df_clean['isOutOfTime']].shape[0]} lignes")

print("\n2.2 GÉNÉRATION DES VARIABLES DE RISQUE...")

X_init_cols = ['age', 'job', 'savingAccountsNum', 'checkingAccountNum', 'creditAmount', 'duration']
df_clean = force_numeric_df(df_clean, X_init_cols)

X_init = df_clean[['age', 'job', 'savingAccountsNum', 'checkingAccountNum', 'creditAmount', 'duration']]
X_init = sm.add_constant(X_init)
df_clean['defaultFlag'] = force_numeric(df_clean['defaultFlag'])
model_init = sm.Logit(df_clean['defaultFlag'], X_init).fit(disp=0)
df_clean['pdInit'] = model_init.predict(X_init)

# Measure of Credit (notation A/B/C) different de MOC C==> conservation

df_clean['moc'] = pd.cut(df_clean['pdInit'], bins=[0, 0.02, 0.05, 1], labels=['A', 'B', 'C'])
df_clean['riskClass'] = pd.qcut(df_clean['pdInit'], q=5, labels=['veryLow', 'low', 'medium', 'high', 'veryHigh'])

print("\n2.3 SÉLECTION DES VARIABLES...")

numericVars = ['age', 'job', 'savingAccountsNum', 'checkingAccountNum', 'creditAmount', 'duration', 'logCreditAmount']
significantVars = []
for var in numericVars:
    var_num = force_numeric(df_clean[var])
    target_num = force_numeric(df_clean['defaultFlag'])
    corr, pval = stats.spearmanr(var_num, target_num)
    if pval < 0.05:
        significantVars.append(var)

def cramersV(x, y):
    confusionMatrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusionMatrix)[0]
    n = confusionMatrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusionMatrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

df_clean_numeric = force_numeric_df(df_clean, significantVars)
corrMatrix = df_clean_numeric[significantVars].corr()
selectedVars = []
for var in significantVars:
    if not any(abs(corrMatrix.loc[var, selected]) > 0.7 for selected in selectedVars):
        selectedVars.append(var)
print(f"  Variables finales sélectionnées : {selectedVars}")



# ============================================================
# PARTIE 3 : MODÈLE IRB-A (BÂLE)
# ============================================================
print("\n" + "="*80)
print("PARTIE 3 : MODÈLE IRB-A (BÂLE) - PD TTC & RWA")
print("="*80)

pdTtcByClass = df_clean.groupby('riskClass')['defaultFlag'].mean()
df_clean['pdTtc'] = df_clean['riskClass'].map(pdTtcByClass)
df_clean['pdTtc'] = force_numeric(df_clean['pdTtc'])

creditAmount_num = force_numeric(df_clean['creditAmount'])
df_clean['onBalanceSheet'] = creditAmount_num * np.random.uniform(0.6, 1.0, len(df_clean))
df_clean['offBalanceSheet'] = creditAmount_num * np.random.uniform(0.1, 0.4, len(df_clean))

def assignCcf(purpose):
    ccfMap = {'car': 0.5, 'business': 0.8, 'education': 0.3, 'furniture/equipment': 0.4, 'radio/TV': 0.2}
    return ccfMap.get(purpose, 0.5)

df_clean['ccf'] = df_clean['purpose'].apply(assignCcf)
ccf_num = force_numeric(df_clean['ccf'])
offBalance_num = force_numeric(df_clean['offBalanceSheet'])
df_clean['ead'] = force_numeric(df_clean['onBalanceSheet']) + ccf_num * offBalance_num

df_clean['lgdBase'] = np.random.uniform(0.25, 0.75, len(df_clean))
df_clean['lgdDownturn'] = np.minimum(1.2 * force_numeric(df_clean['lgdBase']), 0.95)

pdTtc_num = force_numeric(df_clean['pdTtc'])
lgdDownturn_num = force_numeric(df_clean['lgdDownturn'])
df_clean['riskWeight'] = 12.5 * pdTtc_num * lgdDownturn_num
df_clean['riskWeight'] = df_clean['riskWeight'].clip(0.05, 1.0)

ead_num = force_numeric(df_clean['ead'])
riskWeight_num = force_numeric(df_clean['riskWeight'])
df_clean['rwa'] = ead_num * riskWeight_num
df_clean['capitalRequirement'] = df_clean['rwa'] * 0.08
df_clean['capitalBuffer'] = df_clean['rwa'] * 0.025
df_clean['ccybBuffer'] = df_clean['rwa'] * 0.01
df_clean['totalCapitalRequired'] = df_clean['capitalRequirement'] + df_clean['capitalBuffer'] + df_clean['ccybBuffer']

print(f"  RWA total : {df_clean['rwa'].sum():,.0f}")
print(f"  Capital requis total : {df_clean['capitalRequirement'].sum():,.0f}")



# ============================================================
# PARTIE 4 : MODÈLE IFRS 9 - PD PIT, SICR, ECL
# ============================================================
print("\n" + "="*80)
print("PARTIE 4 : MODÈLE IFRS 9 - PD PIT, SICR, ECL")
print("="*80)

meanUnemp = macroData['unemploymentRate'].mean()
stdUnemp = macroData['unemploymentRate'].std()
unemp_num = force_numeric(df_clean['unemploymentRate'])
df_clean['deltaUnemp'] = (unemp_num - meanUnemp) / stdUnemp
beta = 0.6
df_clean['pdPit'] = pdTtc_num * np.exp(beta * force_numeric(df_clean['deltaUnemp']))
df_clean['pdPit'] = df_clean['pdPit'].clip(0.001, 0.999)

df_clean['remainingMonths'] = np.random.randint(1, 60, len(df_clean))
pdPit_num = force_numeric(df_clean['pdPit'])
remaining_num = force_numeric(df_clean['remainingMonths'])
df_clean['pdLifetime'] = 1 - (1 - pdPit_num) ** (remaining_num / 12)

df_clean['pdInit'] = pdTtc_num * np.exp(np.random.normal(0, 0.3, len(df_clean)))
df_clean['pdInit'] = df_clean['pdInit'].clip(0.001, 0.2)
df_clean['mocInit'] = df_clean['pdInit'].apply(lambda x: 'A' if x < 0.02 else ('B' if x < 0.05 else 'C'))

mocMapping = {'A': 1, 'B': 2, 'C': 3}
pdPit_num2 = force_numeric(df_clean['pdPit'])
pdInit_num = force_numeric(df_clean['pdInit'])
relativeIncrease = pdPit_num2 / (pdInit_num + 1e-6)
absoluteIncrease = pdPit_num2 - pdInit_num

mocInitNum = force_numeric(df_clean['mocInit'].map(mocMapping))
mocCurrentNum = force_numeric(df_clean['moc'].map(mocMapping))
mocDowngrade = (mocCurrentNum - mocInitNum) >= 2

df_clean['sicr'] = (relativeIncrease > 1.0) | (absoluteIncrease > 0.005) | mocDowngrade

def updateStaging(row):
    if row['defaultFlag'] == 1:
        return 3
    elif row['sicr']:
        return 2
    else:
        return 1

df_clean['stagingIFRS9'] = df_clean.apply(updateStaging, axis=1)

lgdBase_num = force_numeric(df_clean['lgdBase'])
ead_num2 = force_numeric(df_clean['ead'])
pdPit_num3 = force_numeric(df_clean['pdPit'])
pdLifetime_num = force_numeric(df_clean['pdLifetime'])

df_clean['eclStage1'] = np.where(df_clean['stagingIFRS9'] == 1, pdPit_num3 * lgdBase_num * ead_num2, 0)
df_clean['eclStage2'] = np.where(df_clean['stagingIFRS9'] == 2, pdLifetime_num * lgdBase_num * ead_num2, 0)
df_clean['eclStage3'] = np.where(df_clean['stagingIFRS9'] == 3, pdLifetime_num * lgdBase_num * ead_num2, 0)
df_clean['eclTotal'] = df_clean['eclStage1'] + df_clean['eclStage2'] + df_clean['eclStage3']

print(f"  Taux SICR : {df_clean['sicr'].mean():.2%}")
print(f"  ECL totale : {df_clean['eclTotal'].sum():,.0f}")



# ============================================================
# PARTIE 5 : VALIDATION DU MODÈLE (SKLEARN)
# ============================================================
print("\n" + "="*80)
print("PARTIE 5 : VALIDATION DU MODÈLE")
print("="*80)

train_df = df_clean[~df_clean['isOutOfTime']]
test_df = df_clean[df_clean['isOutOfTime']]


X_train_raw = train_df[selectedVars]
X_test_raw = test_df[selectedVars]

X_train = force_numeric_df(X_train_raw).fillna(0)
X_test = force_numeric_df(X_test_raw).fillna(0)

y_train = pd.to_numeric(train_df['defaultFlag'], errors='coerce').fillna(0).astype(int)
y_test = pd.to_numeric(test_df['defaultFlag'], errors='coerce').fillna(0).astype(int)

print(f"  Train set : {X_train.shape[0]} lignes, {X_train.shape[1]} colonnes")
print(f"  Test set : {X_test.shape[0]} lignes, {X_test.shape[1]} colonnes")
print(f"  Distribution y_train : 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"  Distribution y_test : 0={sum(y_test==0)}, 1={sum(y_test==1)}")


model = LogisticRegression(random_state=42, C=1.0, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

X_oot = force_numeric_df(test_df[selectedVars]).fillna(0)
y_oot = pd.to_numeric(test_df['defaultFlag'], errors='coerce').fillna(0).astype(int)
try:
    oot_pred = model.predict_proba(X_oot)[:, 1]
except:
    oot_pred = model.predict(X_oot).astype(float)

def calculateMetrics(y_true, y_pred, name):

    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    print(f"  {name}: AUC={auc:.4f}, GINI={gini:.4f}, KS={ks:.4f}")
    return {'auc': auc, 'gini': gini, 'ks': ks}
   
  

metrics_train = calculateMetrics(y_train, train_pred, "Train")
metrics_test = calculateMetrics(y_test, test_pred, "Test")
metrics_oot = calculateMetrics(y_oot, oot_pred, "Out-of-Time")

# ============================================================
# PARTIE 6 : STRESS TESTS
# ============================================================
print("\n" + "="*80)
print("PARTIE 6 : STRESS TESTS")
print("="*80)

stressScenarios = {
    'Baseline': {'unempShock': 0.0, 'weight': 0.50},
    'Upside': {'unempShock': -0.20, 'weight': 0.25},
    'Downside': {'unempShock': 0.35, 'weight': 0.25}
}

stressResults = []
for scName, shocks in stressScenarios.items():
    df_stress = df_clean.copy()
    unemp_rate_num = force_numeric(df_stress['unemploymentRate'])
    df_stress['unemploymentStressed'] = unemp_rate_num * (1 + shocks['unempShock'])
    deltaStress = (force_numeric(df_stress['unemploymentStressed']) - meanUnemp) / stdUnemp
    df_stress['pdPitStress'] = force_numeric(df_stress['pdTtc']) * np.exp(beta * deltaStress)
    df_stress['eclStress'] = (force_numeric(df_stress['pdPitStress']) * 
                               force_numeric(df_stress['lgdDownturn']) * 
                               force_numeric(df_stress['ead']))
    
    stressResults.append({
        'Scenario': scName,
        'Shock_Unemployment': shocks['unempShock'],
        'Weight': shocks['weight'],
        'PD_Mean_Stressed': df_stress['pdPitStress'].mean(),
        'ECL_Total': df_stress['eclStress'].sum(),
        'ECL_Variation': (df_stress['eclStress'].sum() / (df_clean['eclTotal'].sum() + 1e-6) - 1) * 100
    })

stressDf = pd.DataFrame(stressResults)

capitalInitial = force_numeric(df_clean['capitalRequirement']).sum()
totalExposure = (force_numeric(df_clean['ead']) * 
                 force_numeric(df_clean['lgdDownturn']) * 
                 12.5 * 0.08).sum()
thresholdPd = capitalInitial / totalExposure if totalExposure != 0 else 0



# ============================================================
# PARTIE 7 : CRÉATION DE LA SORTIE EXCEL 
# ============================================================
print("\n" + "="*80)
print("PARTIE 7 : CRÉATION DE LA SORTIE POUR POWER BI")
print("="*80)

output_file = 'Credit_Risk_sortie.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    columns_to_export = [
        'age', 'sex', 'job', 'housing', 'purpose', 'creditAmount', 'duration',
        'defaultFlag', 'riskClass', 'moc', 'stagingIFRS9', 'sicr',
        'pdInit', 'pdTtc', 'pdPit', 'pdLifetime',
        'ead', 'onBalanceSheet', 'offBalanceSheet', 'ccf',
        'lgdBase', 'lgdDownturn', 'rwa', 'capitalRequirement',
        'eclStage1', 'eclStage2', 'eclStage3', 'eclTotal'
    ]
    available_cols = [col for col in columns_to_export if col in df_clean.columns]
    df_clean[available_cols].head(100).to_excel(writer, sheet_name='Donnees_Clients', index=False)
    
    metrics_summary = pd.DataFrame({
        'Indicateur': [
            'PD TTC moyenne', 'PD PIT moyenne', 'PD Lifetime moyenne',
            'EAD total', 'LGD Downturn moyenne', 'RWA total',
            'Capital requis (8%)', 'Capital Buffer (2.5%)', 'CCyB (1%)',
            'Capital total requis', 'ECL totale IFRS9',
            'Taux SICR', 'Stage 1 (%)', 'Stage 2 (%)', 'Stage 3 (%)',
            'AUC (Train)', 'AUC (Test)', 'AUC (Out-of-Time)',
            'GINI (Out-of-Time)', 'KS (Out-of-Time)',
            'PD seuil de rupture (Reverse Stress)'
        ],
        'Valeur': [
            f"{force_numeric(df_clean['pdTtc']).mean():.4f}",
            f"{force_numeric(df_clean['pdPit']).mean():.4f}",
            f"{force_numeric(df_clean['pdLifetime']).mean():.4f}",
            f"{force_numeric(df_clean['ead']).sum():,.0f}",
            f"{force_numeric(df_clean['lgdDownturn']).mean():.2%}",
            f"{force_numeric(df_clean['rwa']).sum():,.0f}",
            f"{force_numeric(df_clean['capitalRequirement']).sum():,.0f}",
            f"{force_numeric(df_clean['capitalBuffer']).sum():,.0f}",
            f"{force_numeric(df_clean['ccybBuffer']).sum():,.0f}",
            f"{force_numeric(df_clean['totalCapitalRequired']).sum():,.0f}",
            f"{force_numeric(df_clean['eclTotal']).sum():,.0f}",
            f"{df_clean['sicr'].mean():.2%}",
            f"{(df_clean['stagingIFRS9'] == 1).mean():.2%}",
            f"{(df_clean['stagingIFRS9'] == 2).mean():.2%}",
            f"{(df_clean['stagingIFRS9'] == 3).mean():.2%}",
            f"{metrics_train['auc']:.4f}",
            f"{metrics_test['auc']:.4f}",
            f"{metrics_oot['auc']:.4f}",
            f"{metrics_oot['gini']:.4f}",
            f"{metrics_oot['ks']:.4f}",
            f"{thresholdPd:.4f}"
        ]
    })
    metrics_summary.to_excel(writer, sheet_name='SyntheseMetriques', index=False)
    
    stressDf.to_excel(writer, sheet_name='StressTests', index=False)
    
    riskClassDist = df_clean.groupby('riskClass').agg({
        'defaultFlag': ['count', 'mean'],
        'ead': 'sum',
        'eclTotal': 'sum'
    }).round(4)
    riskClassDist.columns = ['Effectif', 'Taux_Defaut', 'EAD_Total', 'ECL_Total']
    riskClassDist = riskClassDist.reset_index()
    riskClassDist.to_excel(writer, sheet_name='Distribution_Risque', index=False)
    
    stagingDist = df_clean.groupby('stagingIFRS9').agg({
        'defaultFlag': ['count', 'mean'],
        'ead': 'sum',
        'eclTotal': 'sum'
    }).round(4)
    stagingDist.columns = ['Effectif', 'Taux_Defaut', 'EAD_Total', 'ECL_Total']
    stagingDist = stagingDist.reset_index()
    stagingDist.to_excel(writer, sheet_name='Distribution_Staging', index=False)
    
    macroData.to_excel(writer, sheet_name='Donnees_Macro', index=False)
    
    if len(selectedVars) > 0 and hasattr(model, 'coef_') and len(model.coef_[0]) == len(selectedVars):
        vars_df = pd.DataFrame({
            'Variables_selectionnees': selectedVars,
            'Coefficients_modele': [float(model.coef_[0][i]) for i in range(len(selectedVars))]
        })
    else:
        vars_df = pd.DataFrame({
            'Variables_selectionnees': selectedVars if len(selectedVars) > 0 else ['Aucune'],
            'Coefficients_modele': [0.0]
        })
    vars_df.to_excel(writer, sheet_name='Variables_Selection', index=False)

print(f"Fichier Excel OK : {output_file}")

print("\n")
print("FIN DU PROJET")
