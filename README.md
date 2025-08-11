# Task 5: Decision Trees and Random Forests

## 📌 Objective
Learn tree-based models for classification & regression, understand overfitting control, interpret feature importances, and evaluate models using cross-validation.

---

## 📂 Dataset
We are using the **Heart Disease Dataset**.  
- **Download Link:** [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Target Variable:** `target` (1 = Disease, 0 = No Disease)

---

## 🛠 Tools & Libraries
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Graphviz (optional for better tree visualization)

---

## 📜 Steps Implemented

### **1. Train a Decision Tree Classifier & Visualize the Tree**
- Loaded dataset and split into train/test sets.
- Trained a Decision Tree classifier.
- Visualized the tree using `plot_tree`.

### **2. Analyze Overfitting & Control Tree Depth**
- Compared performance of a deep (overfitting) tree vs. a shallow (controlled depth) tree.
- Used `max_depth` to prevent overfitting.

### **3. Train a Random Forest & Compare Accuracy**
- Trained a Random Forest classifier with 100 trees.
- Compared accuracy against Decision Tree.

### **4. Interpret Feature Importances**
- Plotted feature importance scores from Random Forest.
- Identified key predictors for heart disease.

### **5. Evaluate Using Cross-Validation**
- Applied 5-fold cross-validation for performance stability.
- Calculated mean accuracy and standard deviation.

---

## 📊 Results Summary

| Model                 | Test Accuracy | Mean CV Accuracy |
|-----------------------|--------------|------------------|
| Decision Tree (Overfit) | ~100% Train / Lower Test | Lower stability |
| Decision Tree (Depth=4) | Improved Test | Higher stability |
| Random Forest         | Higher Test Accuracy | Stable results |

---

## 📌 Interview Questions & Answers

**Q1:** What is a Decision Tree?  
A: A supervised ML model that splits data into rules for predictions.

**Q2:** How does `max_depth` affect performance?  
A: Large depth → overfitting; small depth → underfitting.

**Q3:** What is Random Forest?  
A: An ensemble of Decision Trees trained on random subsets of data & features.

**Q4:** Why does Random Forest perform better than a single tree?  
A: It reduces variance by averaging multiple models.

**Q5:** What is feature importance?  
A: A score that shows how much each feature influences predictions.

---

## 📌 What You'll Learn
- Decision tree fundamentals.
- Controlling model complexity.
- Benefits of ensemble methods.
- Reading feature importances.
- Reliable evaluation with cross-validation.

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python decision_tree_random_forest.py
