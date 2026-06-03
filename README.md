## Résumé technique

| Composant | Contenu |
|-----------|---------|
| **Données** | Fichier Excel `german_credit_data.xlsx` (1000 clients, 10 attributs) + données macro synthétiques (chômage, PIB, taux, HPI) |
| **Nettoyage** | Création de `defaultFlag`, encodage des variables qualitatives, division en périodes **in‑sample** (2000‑2016) et **out‑of‑time** (2017‑2019) |
| **PD TTC** | Probabilité de défaut *Through The Cycle* par classe de risque (veryLow → veryHigh) |
| **PD PIT** | Ajustement cyclique via un modèle de type `PD_TTC * exp(β × Δchômage)` avec β = 0,6 |
| **PD Lifetime** | `1 - (1 - PD_PIT)^(durée restante en années)` pour les stades 2 et 3 |
| **EAD** | `On‑Balance + CCF × Off‑Balance`, avec CCF selon la finalité du crédit |
| **LGD** | `LGD_base` aléatoire (25‑75 %) et `LGD_downturn = min(1,2 × LGD_base ; 0,95)` |
| **RWA / Capital** | `RWA = EAD × 12,5 × PD_TTC × LGD_downturn` ; capital = RWA × (8 % + 2,5 % + 1 %) |
| **SICR** | Détection si : PD actuelle > 2 × PD initiale **OU** +0,5 point **OU** dégradation de 2 crans (MoC : A→C) |
| **ECL** | Stage 1 : `PD_PIT × LGD_base × EAD` ; Stages 2 & 3 : `PD_Lifetime × LGD_base × EAD` |
| **Validation** | Régression logistique avec AUC / GINI / KS sur train, test et OOT |
| **Stress tests** | Scénarios Upside (-20 % chômage), Baseline (0 %), Downside (+35 %) – variation des ECL |
| **Reverse stress** | `PD_seuil = Capital_initial / (EAD × LGD_downturn × 12,5 × 8 %)` |
| **Exports** | Fichier Excel `Credit_Risk_sortie.xlsx` (7 feuilles) + code M/DAX pour Power BI |

Le projet illustre concrètement la **double logique** :  
- **Bâle** → fonds propres stables (PD TTC, LGD downturn, RWA)  
- **IFRS 9** → provisions réactives (PD PIT, PD Lifetime, staging)
