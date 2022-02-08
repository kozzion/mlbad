import pandas as pd
import numpy as np
from svd_bad import SVDBad

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

#https://www.kaggle.com/mitishaagarwal/patient
df = pd.read_csv(r'dataset.csv')
list_name_column = np.array(list(df.columns))
list_dtype_column = list(df.dtypes)
array_data = df.to_numpy()

# print(array_data.shape)
# df.dropna(axis='rows')
# array_data = df.to_numpy()
# print(array_data.shape)

array_column_input = np.array([False] * len(list_name_column))
array_column_output = np.array([False] * len(list_name_column))

array_column_input[22:71] = True

array_column_output[75] = True # diabetes_mellitus 
array_column_output[76] = True # hepatic_failure 
array_column_output[78] = True # leukemia
array_column_output[79] = True # lymphoma 
array_column_output[84] = True # death

# print(list_name_column[array_column_input])
# print(list_name_column[array_column_output])
# print(array_data.shape)
# for i, name_column, dtype_column  in zip(range(len(list_name_column)), list_name_column, list_dtype_column):
#     print(i)
#     print(name_column)
#     print(dtype_column)

array_input = array_data[:,array_column_input].astype('float64')
array_output_true = array_data[:,array_column_output].astype('float64')
array_input_row_clean = ~np.isnan(array_input).any(axis=1)
array_output_row_clean = ~np.isnan(array_output_true).any(axis=1)
array_row_clean = array_input_row_clean & array_output_row_clean

array_input = array_input[array_row_clean, :]
array_output_true = array_output_true[array_row_clean, :]

svd = SVDBad()
svd.fit(array_input, array_output_true)
array_output_pred = svd.transform(array_input)
array_input_pred = svd.inverse_transform(array_output_pred)

error_output = np.mean(array_output_true - array_output_pred , axis=0)
error_input = np.mean(array_input - array_input_pred, axis=0)
print(error_output.shape)
print(error_output / np.mean(array_output_true, axis=0) )
print(error_input / np.mean(array_input, axis=0) )

fpr, tpr, _ = roc_curve(array_output_true[:,4], array_output_pred[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()