import pandas as pd
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

features = pd.read_csv('./data/features.csv', index_col='match_id')
features = features.fillna(value=0)

y_train = features['radiant_win']
X_train = features.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                         'barracks_status_radiant', 'barracks_status_dire', 'start_time'], axis=1)


kf = KFold(n_splits=5, shuffle=True,  random_state=42)
c_arr = [1, 4, 16, 64, 256, 1024]

def run_log_regression(X, y):
    for penalty_scale in c_arr:
        start_time = datetime.datetime.now()
        log_regr = LogisticRegression(random_state=241, C=penalty_scale, penalty='l2')
        log_regr.fit(X, y)
        cv_score = cross_val_score(log_regr, X=X, y=y, cv=kf, scoring='roc_auc').mean()
        print('#penalty_scale:', penalty_scale, "roc-auc score: ", cv_score, 'Time elapsed:', datetime.datetime.now() - start_time)


# X_train_scaled = preprocessing.StandardScaler().fit_transform(X_train)
# print('1. no preprocessing')
# run_log_regression(X_train_scaled, y_train)

hero_columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
cat_columns = hero_columns.copy()
cat_columns.append('lobby_type')

X_train_no_cat = X_train.drop(cat_columns, axis=1)
X_train_scaled = preprocessing.StandardScaler().fit_transform(X_train_no_cat)
# print('2. exclude categories')
# run_log_regression(X_train_scaled, y_train)

# all_heroes = []
# for hc in hero_columns:
#     all_heroes.extend(X_train[hc])
#
# distinct_heroes_used = len(pd.Series(all_heroes).unique()) # which is 108
distinct_heroes_count = 113 # see heroes.csv, this is the source of confusion
hero_id_participation = []
for i in range(distinct_heroes_count):
    hero_id_participation.append("participate_of_hero_"+str(i + 1))
X_pick = pd.DataFrame(0, index=X_train.index, columns=hero_id_participation)
#
for i, match_id in enumerate(X_train.index):
    for p in range(5):
        radian_hero = X_train.loc[match_id, 'r%d_hero' % (p+1)]
        dire_hero = X_train.loc[match_id, 'd%d_hero' % (p+1)]
        X_pick.iloc[i, radian_hero - 1] = 1
        X_pick.iloc[i, dire_hero - 1] = -1
#
X_train_mapped_cat = pd.concat([X_train_no_cat, X_pick], axis=1)
X_train_scaled = preprocessing.StandardScaler().fit_transform(X_train_mapped_cat)
# print('3. mapped categories')
# run_log_regression(X_train_scaled, y_train)


features_test = pd.read_csv('./data/features_test.csv', index_col='match_id')
features_test = features_test.fillna(value=0)
X_test = features_test.drop(['start_time'], axis=1)

X_pick_test = pd.DataFrame(0, index=X_test.index, columns=hero_id_participation)

for i, match_id in enumerate(X_test.index):
    for p in range(5):
        radian_hero = X_test.loc[match_id, 'r%d_hero' % (p+1)]
        dire_hero = X_test.loc[match_id, 'd%d_hero' % (p+1)]
        X_pick_test.iloc[i, radian_hero - 1] = 1
        X_pick_test.iloc[i, dire_hero - 1] = -1

X_test = X_test.drop(cat_columns, axis=1)
X_test = pd.concat([X_test, X_pick_test], axis=1)
X_test_scaled = preprocessing.StandardScaler().fit_transform(X_test)
log_regr_best = LogisticRegression(random_state=241, penalty='l2')
log_regr_best.fit(X_train_scaled, y_train)
y_predict = log_regr_best.predict_proba(X_test_scaled)[:, 1]
print(pd.Series(y_predict).max())
print(pd.Series(y_predict).min())

# Какое качество получилось у логистической регрессии над всеми исходными признаками?
# Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу?
# Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

# #penalty_scale: 1 roc-auc score:  0.7191894167083216 Time elapsed: 0:00:03.710110

# Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)?
# Чем вы можете объяснить это изменение?

# Сколько различных идентификаторов героев существует в данной игре?
# 113

# Какое получилось качество при добавлении "мешка слов" по героям?
# Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
# 0.754 Time elapsed: 0:00:08.18

# Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
# 0.9966521921146925
# 0.008615583197616324