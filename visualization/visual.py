from sklearn.preprocessing import LabelEncoder , StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("cleaned.csv")

numcolums=["Monthly Income","Rent/Housing","Food & Groceries","Entertainment & Shopping","Travel & Transport","Savings/Investments"]

ss = StandardScaler()

df[numcolums] = ss.fit_transform(df[numcolums])



X = df[numcolums]
y = df["Personalized Recommendation"]

le = LabelEncoder()

y_encoded = le.fit_transform(y)

X_train , X_test , y_train , y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=42)

model1 = XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
model1.fit(X_train,y_train)

model2 = RandomForestClassifier(n_estimators=100,random_state=42)
model2.fit(X_train,y_train)

y_pred1 = model1.predict(X_test)

y_pred2 = model2.predict(X_test)

cr1 = classification_report(y_test,y_pred1)
cr2 = classification_report(y_test,y_pred2)

cm1 = confusion_matrix(y_test,y_pred1)
cm2 = confusion_matrix(y_test,y_pred2)

# print(cr1)
# print(cr2)
# print(cm1)
# print(cm2)

# plt.figure(figsize=(6,7))
# sns.heatmap(cm1,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Fail","Pass"],yticklabels=["Pass","Fail","Fail"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confussion Matrix")
# plt.tight_layout()
# plt.show()


with open("scalar.pkl","wb") as f:
    pickle.dump(ss,f)