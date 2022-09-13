
# Importar libreria

from pyexpat import model
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

df_train=pd.read_csv("data/train.csv",sep=",",index_col=[0])
df_test=pd.read_csv("data/test.csv",sep=",",index_col=[0])


X_train=df_train.drop(columns="Credit_Score")
X_test=df_test.drop(columns="Credit_Score")
y_train=df_train["Credit_Score"]
y_test=df_test["Credit_Score"]

print(X_train.shape)
print(X_test.shape)

rfc_model = RandomForestClassifier(random_state=7,n_estimators=322,max_depth=38,max_leaf_nodes=11000,oob_score=True,max_features=4)
rfc_model.fit(X_train, y_train)

rfc_model_pred = rfc_model.predict(X_test)
print(classification_report(y_test, rfc_model_pred))

print(accuracy_score(y_test,rfc_model_pred))

fecha = datetime.datetime.now().strftime('%Y%m%d%H:%M:%S')
with open('model'+ str(fecha)+'.model' , 'wb') as archivo_salida:
    pickle.dump(rfc_model, archivo_salida)

# En esta parte probamos en importar el mejor modelo:

print("............................")

with open('model/my_model.model', "rb") as archivo_entrada:
    modelo_importada = pickle.load(archivo_entrada)

print(modelo_importada)

modelo_importada_pred = modelo_importada.predict(X_test)

print(classification_report(y_test, modelo_importada_pred))










