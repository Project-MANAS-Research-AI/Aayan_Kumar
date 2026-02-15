import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("gene_expression.csv")
x = df.drop(columns='Cancer Present').values
y = df['Cancer Present'].to_numpy().copy()
y[y == 0] = -1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
def train(x,y,learning_rate=0.01,C=1,epoch=1000):
    m,n=x.shape
    lam=1/C
    w=np.zeros(n)
    b=0.0
    for e in range(epoch):
        for i in range(m):
            pred=y[i]*(np.dot(w,x[i])+b)
            if pred>=1:
               w=w-learning_rate*lam*w
            else:
                w=w-learning_rate*(w*lam-y[i]*x[i])
                b=b+learning_rate*y[i]
        if e%100==0:
            loss=0.5*lam*(w@w)+np.mean(np.maximum(0,1-y*(x@w+b)))
            print(f"Epochs: {e} | Loss: {loss}")
    return w,b
def pred(x,w,b):
    return np.dot(x,w)+b
def accuracy(pred,y):
    signs=np.sign(pred)
    accu=np.mean(signs==y)
    return accu

#implementing
w_op,b_op=train(x_train,y_train,epoch=100,C=5)
prediction = pred(x_test, w_op, b_op)
ac=accuracy(prediction,y_test)
print("Accuracy: ",ac)
