# HR Analytics – Predicting Employee Attrition

## Introduction
Employee attrition impacts cost, productivity, and morale. We analyze HR data to identify key drivers and predict attrition risk, enabling proactive retention.

## Abstract
We perform EDA, build classification models (Logistic Regression, Decision Tree), evaluate performance, and explain predictions with SHAP. A Power BI dashboard surfaces risk segments (department, age, income, overtime). Final recommendations target high-impact levers like overtime and tenure.

## Tools Used
Python (Pandas, NumPy, Seaborn, Scikit-learn), SHAP, Power BI.

## Steps Involved
1. **Data Preparation**: cleaned nulls/duplicates, removed ID-like columns.
2. **EDA**: class balance, attrition by department/age/income, correlation map.
3. **Modeling**: train/test split (80/20), Logistic Regression & Decision Tree with balanced classes.
4. **Evaluation**: accuracy, precision, recall, F1; confusion matrices.
5. **Explainability**: SHAP summary plot to rank drivers.
6. **Dashboard**: `clean_hr_data.csv` imported into Power BI with slicers (Gender, Education, OverTime).

## Results (sample placeholders)
- Best model: Decision Tree (Accuracy: __, F1: __)
- Key drivers: OverTime ↑, MonthlyIncome ↓, YearsAtCompany ↕, JobRole, Age.
- At-risk segments: [e.g., Sales, 0–2 yrs tenure, overtime=Yes]

## Recommendations
- Optimize overtime policies; monitor workloads.
- Early-tenure mentorship program (0–2 years).
- Compensation bands review for roles with high income-related attrition.
- Career pathing: regular promotions for mid-tenure employees.

## Conclusion
The model + dashboard provide actionable visibility into attrition risk, enabling targeted retention initiatives.

## Appendix (links)
- Metrics: `artifacts/tables/metrics.csv`
- Confusion matrices: `artifacts/tables/confusion_matrices.csv`
- SHAP plot: `artifacts/figures/shap_summary.png`
