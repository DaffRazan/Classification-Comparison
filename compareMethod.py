# import sk learn dan pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# import dataset
data = pd.read_csv('magicGammaTelescope.csv')

# memisahkan antara data dan label
X = data.iloc[:, :-1].values
y = data.iloc[:, len(data.columns)-1].values

# split data training dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# scaling data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
# model = SVC(kernel='rbf')
# model.fit(X_train, y_train)

# KNN
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train, y_train)
### end of model #####
######################

# hasil prediksi
hasil_prediksi_svm = model.predict(X_test)
print(hasil_prediksi_svm)

# confusion matrix
cm = confusion_matrix(y_test, hasil_prediksi_svm)
print(cm)

# classification report (precision,recall,F1-score,accuracy,macro avg,dan weighted avg)
print(classification_report(y_test, hasil_prediksi_svm))
