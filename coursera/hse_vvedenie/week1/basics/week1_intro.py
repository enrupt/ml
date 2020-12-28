import pandas as pd
import re

df = pd.read_csv('../titanic.csv', index_col='PassengerId')

total_count = len(df.index)
male_count = df[df['Sex'] == 'male']['Sex'].count()
female_count = total_count - male_count
print(str(male_count) + " " + str(female_count))

survied = df[df['Survived'] == 1]['Survived'].count()
print("{:0.2f}".format(survied * 100 / total_count))

first_class = df[df['Pclass'] == 1]['Pclass'].count()
print("{:0.2f}".format(first_class * 100 / total_count))

print(df['Age'].agg(['mean', 'median']).round(1))

SibSp = df['SibSp']
Parch = df['Parch']
correlation = SibSp.corr(Parch)
print("{:0.2f}".format(correlation))


def name_etractor(str):
    name_in_braces_res = re.search('.*\\((.*)\\).*', str)
    if name_in_braces_res:
        name_in_braces = name_in_braces_res.group(1)
        return name_in_braces.split(' ')[0]

    miss_list = str.split('Miss.')
    if len(miss_list) < 2:
        #print("Unexpected str "+str)
        return str
    return miss_list[1].strip()

female_df = df[df['Sex'] == 'female']['Name'].apply(name_etractor)
print(female_df.value_counts())

# print("{:0.2f}".format(corr))
