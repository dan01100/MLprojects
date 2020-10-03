import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


fighters = pd.read_csv("fight_data/fighters.csv", index_col="Name")
del fighters['Unnamed: 0']

bouts = pd.read_csv("fight_data/Bouts.csv", index_col=0)

fighters["win_rate"] = fighters["Wins"] / (fighters["Wins"] + fighters["Loses"])


#Creates  2 vectors (winVec - loseVec & loseVec - winVec)
def toVec(winner, loser, fighter_data):
 
    if winner not in fighter_data.index or loser not in fighter_data.index:
        return None
     
    winVec = fighter_data.loc[winner].to_numpy()
    loseVec = fighter_data.loc[loser].to_numpy()
    
    
    out = winVec - loseVec
    out2 = loseVec - winVec
    
    
    return (out, out2)


def createAllVecs(bout_data, fighter_data):
    
    vectors = []
    labels = []
    

    for i, j in bout_data.iterrows(): 
        
        
        winner = j["win"]
        loser = j["lose"]
        
        x = toVec(winner, loser, fighter_data)     
        
        if x is None:
            continue

        out, out2 = x
        
        vectors.append(out)
        labels.append(1)
        vectors.append(out2)
        labels.append(0)
                
    
    return (vectors, labels)

X, y = createAllVecs(bouts.head(3000), fighters)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)


#Normalize data
std_scale = StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)


model = XGBClassifier()
model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.001,  
                      colsample_bytree = 0.3,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=6, 
                      gamma=1)


model.fit(np.asarray(X_train), y_train)

y_pred = model.predict(np.asarray(X_test))

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


