import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

Border='-'*40
print(Border)
print("step 1 : load the dataset :")
print(Border)

df=pd.read_csv("Employee_Attrition.csv")

print("Data loaded successfully")

print(Border)
print("step 2 : Data analysis")
print(Border)

print("shape of dataset is :")
print(df.shape)

print("Column in dataset :")
print(df.columns)

print("First five record is :")
print(df.head())

print(Border)
print("step 3 : Check missing value :")
print(Border)

print("missing value is :")
print(df.isnull().sum())

print(Border)
print("step 4 : Convert categorical feature into numerical representation :")
print(Border)

df["OverTime"]=df["OverTime"].map({"Yes":1,"No":0})

df["Attrition"]=df["Attrition"].map({"Yes":1,"No":0})

print("Conversion sucessfully ")

print(Border)
print("step 5 : separate X and Y :")
print(Border)

X=df.drop("Attrition",axis=1)
Y=df["Attrition"]

print(Border)
print("step 6 : split the dataset training and testing :")
print(Border)

X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print(Border)
print("step 7 : Feature Scaling :")
print(Border)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print(Border)
print("step 8 : Create MLP model :")
print(Border)

model=MLPClassifier(
    hidden_layer_sizes=(10,8),
    max_iter=1000,
    random_state=42
)

print("Model Create successfully")

print(Border)
print("step 9 : Train the model:")
print(Border)

model =model.fit(X_train,Y_train)

print(Border)
print("step 10 : Number of iteration :")
print(Border)

print("\n Iteration:",model.n_iter_)

print(Border)
print("step 11 :Training Accuracy :")
print(Border)

Train_prediction=model.predict(X_train)
train_accuracy=accuracy_score(Y_train,Train_prediction)

print("Training accuracy is :",train_accuracy*100,"%")

print(Border)
print("step 12 :Testing Accuracy :")
print(Border)

Y_pred=model.predict(X_test)
test_accuracy=accuracy_score(Y_test,Y_pred)

print("Testing accuracy is :",test_accuracy*100,"%")

print(Border)
print("step 13 :Confusion matrix :")
print(Border)

cm=confusion_matrix(Y_test,Y_pred)
print(cm)

print(Border)
print("step 14 :Loss curve :")
print(Border)

plt.plot(model.loss_curve_)
plt.title("MLP loss curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

print(Border)
print("step 15 :Prediction Function:")
print(Border)

def PredictAttrition(employee_data):

    data=pd.DataFrame([employee_data], columns=X.columns)

    data["OverTime"]=data["OverTime"].map({"Yes":1,"No":0})

    data=scaler.transform(data)

    result=model.predict(data)

    if result[0]==1:
        print("Employee is likely to LEAVE")
    else:
        print("Employee is likely to STAY")

PredictAttrition([25,30000,2,3,10,2,2,"Yes",3,2])