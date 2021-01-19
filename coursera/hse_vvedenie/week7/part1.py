import pandas
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

#
# Подход 1 - код
#

features = pandas.read_csv('./data/features.csv', index_col='match_id')
total_count = len(features.index)
print("have misses:")
for column in features.columns:
    count_col = features[column].count()
    if count_col != total_count:
        print(column)

features = features.fillna(value=0)

y_train = features['radiant_win']
X_train = features.drop(['radiant_win', 'tower_status_radiant', 'tower_status_dire',
                   'barracks_status_radiant', 'barracks_status_dire'], axis=1)

features_test = pandas.read_csv('./data/features_test.csv', index_col='match_id')


kf = KFold(n_splits=5, shuffle=True,  random_state=42)
trees_num_arr = [10, 20, 30, 40, 50, 100]

for trees_num in trees_num_arr:
    start_time = datetime.datetime.now()
    gbcl = GradientBoostingClassifier(n_estimators=trees_num, verbose=False, random_state=241)
    gbcl.fit(X_train, y_train)
    cv_score = cross_val_score(gbcl, X=X_train, y=y_train, cv=kf, scoring='roc_auc').mean()
    print('#trees:', trees_num, "roc-auc score: ", cv_score, 'Time elapsed:', datetime.datetime.now() - start_time)

#
# Подход 1 - отчет
#
# B: Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
# O: следующе признаки имеют NaN-ы

# объясняется это, по всей видимости, тем,
# что за первые 5 минут не произошло соответствующего события

# B: Как называется столбец, содержащий целевую переменную?
# О: radiant_win

# B: Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.
# О:


# В: Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
# О: 