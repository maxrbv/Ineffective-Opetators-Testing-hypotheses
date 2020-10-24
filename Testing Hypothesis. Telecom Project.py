import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats import weightstats as stests

# STEP 1. General information
data = pd.read_csv('telecom_dataset_us.csv')
print(data.info(), '\n')
print(data.head(), '\n')

# STEP 2. Data preprocessing
# missing values
print('Missing values:')
print(data.isnull().sum(), '\n')
data.dropna(inplace=True)

# drop duplicate values
data = data.drop_duplicates()

# converting data types
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['date'] = data['date'].dt.date.astype('datetime64[ns]')
data['operator_id'] = data['operator_id'].astype('int64')
data['internal'] = data['internal'].astype(bool)

# STEP 3. EDA
print('Numerical values info:')
print(data.drop(['user_id', 'operator_id'], axis=1).describe())

# plotting histograms
columns = ['calls_count', 'call_duration', 'total_call_duration']
for column in columns:
    title = str(column) + ' histogram'
    plt.figure(figsize=(18, 7))
    sns.distplot(data[column]).set_title(title)
    plt.show()


# removing outliers
def subset_by_iqr(df, column):
    """Remove outliers from a dataframe by column, removing rows for which the column value are
       less than Q1-1.5IQR or greater than Q3+1.5IQR.
    Args:
        df (`:obj:pd.DataFrame`: A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    # calculating Q1, Q3, IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # applying filter with respect to IQR
    df_filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df.loc[df_filter]


data = subset_by_iqr(data, 'calls_count')
data = subset_by_iqr(data, 'call_duration')
data = subset_by_iqr(data, 'total_call_duration')

# new histograms after using IQR method
for column in columns:
    title = str(column) + ' histogram'
    plt.figure(figsize=(18, 7))
    sns.distplot(data[column]).set_title(title)
    plt.show()

# STEP 4. Ineffective operators
# FIRST CRITERIA
# total calls and missed calls ratio for each operator
total_calls = data.groupby(by='operator_id', as_index=False).agg({'calls_count':'sum', 'is_missed_call':'mean'}).sort_values(by='is_missed_call', ascending=False)
total_calls.columns = ['operator_id', 'total_calls_count', 'missed_calls_ratio']

# histograms
plt.figure(figsize=(18, 7))
sns.distplot(total_calls['total_calls_count'])
plt.show()
plt.figure(figsize=(18, 7))
sns.distplot(total_calls['missed_calls_ratio'])
plt.show()

# ineffective operatrs by total_calls
threshold_calls_count = np.percentile(total_calls['total_calls_count'], 50)
ineffective_operators_calls = total_calls.query('total_calls_count >= @threshold_calls_count')

# ineffective operators by missed_calls_ratio
threshold_missed_calls = np.percentile(total_calls['missed_calls_ratio'], 90)
ineffective_operators_missed_calls = ineffective_operators_calls.query('missed_calls_ratio >= @threshold_missed_calls')

# ineffective operators to list
ineffective_operators_list = ineffective_operators_missed_calls['operator_id'].to_list()

# SECOND CRITERIA
# waiting time
data['waiting_time'] = data['total_call_duration'] - data['call_duration']

# total calls and mean waiting time
data_in = data.query('direction == "in"')
waiting_calls = data_in.groupby(by='operator_id', as_index=False).agg({'calls_count':'sum', 'waiting_time':'mean'})
waiting_calls.columns = ['operator_id', 'total_calls_count', 'mean_waiting_time']

# histograms
plt.figure(figsize=(18, 7))
sns.distplot(waiting_calls['total_calls_count'])
plt.show()
plt.figure(figsize=(18, 7))
sns.distplot(waiting_calls['mean_waiting_time'])
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
plt.show()

# ineffective operatrs by total_calls
ineffective_operators_calls = waiting_calls.query('total_calls_count >= @threshold_calls_count')

# ineffective operators by mean waiting time
threshold_waiting_time = np.percentile(waiting_calls['mean_waiting_time'], 90)
ineffective_operators_waiting = ineffective_operators_calls.query('mean_waiting_time >= @threshold_waiting_time')

# adding ineffective operators to list
ineffective_operators_list += ineffective_operators_waiting['operator_id'].to_list()

# THIRD CRITERIA
outgoing_calls = data.query('direction == "out"').groupby(by='operator_id', as_index=False).agg({'calls_count':'sum'})
outgoing_calls.columns = ['operator_id','outgoing_calls_count']
total_calls = data.groupby(by='operator_id', as_index=False).agg({'calls_count':'sum'})
total_calls.columns = ['operator_id', 'total_calls_count']

outgoing_total_calls = pd.merge(outgoing_calls, total_calls, how='inner', on='operator_id')

outgoing_total_calls['outgoing_total_calls_ratio'] = outgoing_total_calls['outgoing_calls_count'] / outgoing_total_calls['total_calls_count']
# print(outgoing_total_calls)

plt.figure(figsize=(18, 7))
sns.distplot(outgoing_total_calls['outgoing_total_calls_ratio'])
plt.show()

ineffective_operators_calls = outgoing_total_calls.query('total_calls_count >= @threshold_calls_count')

threshold_outgoing_calls = np.percentile(outgoing_total_calls['outgoing_total_calls_ratio'], 10)
ineffective_operators_outgoing = ineffective_operators_calls.query('outgoing_total_calls_ratio <= '
                                                                   '@threshold_outgoing_calls')

ineffective_operators_list += ineffective_operators_outgoing['operator_id'].to_list()

# Result list
ineffective_operators_array = np.array(ineffective_operators_list)
ineffective_operators_array = np.unique(ineffective_operators_array)

# STEP 5. Testing hypotheses
# 1. Does the average call duration for ineffective operators differ from other operators?
print('Null hypothesis:"There is no difference in average call_duration between ineffective and other operators". The '
      'alternative hypothesis:"There is a difference in average call_duration between ineffective and other '
      'operators".')
# information about ineffective operators
ineffective_data = data.query('operator_id in @ineffective_operators_array')
# information about random effective operators
rnd_effective_data = data.query('operator_id not in @ineffective_operators_array').sample(n = ineffective_data.shape[0], random_state = 1)

alpha = .05

ztest, p_val = stests.ztest(ineffective_data['call_duration'], x2=rnd_effective_data['call_duration'])
print('pvalue is:', float(p_val))
if p_val < alpha:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
print()

# 2. Does the average waiting time for A tariff users differ from average waiting time for B tariff users?
print('Null hypothesis:"There is no difference in average waiting_time between A tariff users and B tariff users". '
      'The alternative hypothesis:"There is a difference in average waiting_time between A tariff users and B tariff '
      'users"')
tariffs = pd.read_csv('telecom_clients_us.csv')

a_users = tariffs.query('tariff_plan == "A"')
b_users = tariffs.query('tariff_plan == "B"')

a_users_id = a_users['user_id'].to_list()
b_users_id = b_users['user_id'].to_list()

a_users_data = data.query('user_id in @a_users_id')
b_users_data = data.query('user_id in @b_users_id').sample(n=a_users_data.shape[0], random_state = 1)

alpha = .05

ztest, p_val = stests.ztest(a_users_data['waiting_time'], x2=b_users_data['waiting_time'])
print('pvalue is:', float(p_val))
if p_val < alpha:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


