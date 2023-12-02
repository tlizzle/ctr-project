import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




"""
dependent var = click: 0/1 for non-click/click
independent vars = 
    id: ad identifier
    hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    C1 -- anonymized categorical variable
    banner_pos: ads postion
    site_id
    site_domain
    site_category
    app_id
    app_domain
    app_category
    device_id
"""


'''loading data
'''
_data_path = './avazu-ctr-prediction/'
_train_data = os.path.join(_data_path, 'train.gz') 
df = pd.read_csv(_train_data, compression = 'gzip', nrows= 1000000)
unused_cols = ["id"]
df.drop(unused_cols, axis= 1, inplace= True)



df = df.sample(frac = 1, random_state= 1)
df = df.sample(n = 10000000, random_state = 1)
df = df.sample(frac = 1, random_state= 1)
df = df.sample(n = 8000000, random_state = 1)
df = df.sample(frac = 1, random_state= 1)
df = df.sample(n = 6000000, random_state = 1)
df = df.sample(frac = 1, random_state= 1)
df = df.sample(n = 4000000, random_state = 1)
df = df.sample(frac = 1, random_state= 1)
df = df.sample(n = 2000000, random_state = 1)


# convert raw date formate to official fromat
df['hour'] = df['hour'].astype(str).apply(lambda x: pd.datetime.strptime(x, '%y%m%d%H'))



''' data exploration
'''
df.shape
df.head()
df.columns

print(df.click.value_counts() / len(df))


# check no duplicates
df.groupby('click').size().describe()

# check for null values
df.isnull().values.any()
print(df.isnull().sum())
print(df.info())


''' data visulization 
'''
# check the proportion of labels
print(sns.countplot(x= 'click', data= df))
print(df['hour'].describe())
df.groupby('hour').agg({'click': 'sum'}).plot(figsize = (13,6), grid= True)
plt.ylabel('click rate')
### starting from 2014/10/21 to 2014/10/31 and the peak is around 22-23 and 28-29

df['time'] = df.hour.apply(lambda x: x.hour)
df.groupby('time').agg({'click': 'sum'}).plot(figsize= (12,6), grid= True)
plt.ylabel('num of click')
plt.title('Time and its coressponding volume')
### the peak of Num of click is between 13:00 to 14:00


df.groupby(['time', 'click']).size().unstack().plot(kind= 'bar', figsize= (13, 6), grid= True)
plt.ylabel('Non-click and click volumne')



df_hour = df[['time','click']].groupby(['time']).count().reset_index()
df_hour = df_hour.rename(columns={'click': 'impressions'})

df_click = df[df['click'] == 1]
df_hour['clicks'] = df_click[['time','click']].groupby(['time']).count().reset_index()['click']

df_hour['CTR'] = df_hour['clicks'] / df_hour['impressions'] 
df_hour.head()

plt.figure(figsize= (12,6))
sns.barplot(y='CTR', x='time', data=df_hour)
plt.title('Click rate distribution')
### found the CTR is high at 1, 7, 15 (13 not the highest point) but not including 13:00 which found previously.
### this implies that high num of click doesnt quarantee high CRT



df['weekday'] = pd.to_datetime(df.hour).dt.dayofweek.apply(lambda x: x+1).astype(str)
df.groupby('weekday').agg({'click':'sum'}).plot(figsize=(12,6))
ticks = list(range(0, 7, 1)) 
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.xticks(ticks, cats)
plt.title('weekday volume')



df.groupby(['weekday','click']).size().unstack().plot(kind='bar', title="Day of the Week", figsize=(12,6))
ticks = list(range(0, 7, 1)) 
plt.xticks(ticks, cats)
plt.title('weekdays distribution')
### can tell tuesday got the highest total volume and clicks


df_dayofweek = df[['weekday','click']].groupby(['weekday']).count().reset_index()
df_dayofweek = df_dayofweek.rename(columns={'click': 'impressions'})
df_click = df[df['click'] == 1]
df_dayofweek['clicks'] = df_click[['weekday','click']].groupby(['weekday']).count().reset_index()['click']
df_dayofweek['CTR'] = df_dayofweek['clicks']/ df_dayofweek['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='weekday',data=df_dayofweek)
plt.title('weekdata distribution')
### again, high num of click does not qurantee high CTR. In fact, Sat and Sun got the highest CTR which makes
### sense that people tend to got more spare time and leading the increase on shoppinh (also increase CTR)


banner_pos = df.banner_pos.unique()
banner_pos.sort()
for i in banner_pos:
    click_mean = df.loc[np.where((df.banner_pos == i))].click.mean()
    print(" banner pos: {},  click avg: {}".format(i, click_mean))
df.groupby(['banner_pos', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='banner position')

df_banner = df[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()
df_banner = df_banner.rename(columns={'click': 'impressions'})
df_banner['clicks'] = df_click[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()['click']
df_banner['CTR'] = df_banner['clicks']/ df_banner['impressions']
sort_banners = df_banner.sort_values(by='CTR',ascending=False)['banner_pos'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y= 'CTR', x= 'banner_pos', data= df_banner, order= sort_banners)
plt.title('Distribution related to banner position')
### 位置0跟1 的點擊總數都是最多的但他們的點擊率並非最高



df[['device_type','click']].groupby(['device_type','click']).size().unstack().plot(kind='bar', title= 'device type')
df_click[df_click['device_type']==1].groupby(['time', 'click'])\
                                     .size().unstack()\
                                     .plot(kind='bar', title= "device 1 click volumne", figsize=(12,6))

### 這樣看起來設備的種類也會對response var 有所影響
### 前張圖發現設備1 的展現量與觸及量都最明顯的，所以之後的圖是特別要看type1 詳細資料 ，按照不同時間區段來觀察其資料變化


app_id_unique = df.app_id.nunique()
app_domain_unique = df.app_domain.nunique()
app_category_unique = df.app_category.nunique()
print(f'Unique value for app_id is {app_id_unique}')
print(f'Unique value for app_domain is {app_domain_unique}')
print(f'Unique value for app_category is {app_category_unique}')

df.app_category.value_counts() / len(df)
df['app_category'].value_counts().plot(kind= 'bar', title= 'clicks for app_category', figsize = (13,5))
### 主要看 app id 各個group 的比例，這裡可以考慮把從第6個位置一直到最後都給同樣category 值


site_id_unique = df.site_id.nunique()
site_domain_unique = df.site_domain.nunique()
site_category_unique = df.site_category.nunique()
device_id_unique = df.device_id.nunique()
device_ip_unique = df.device_ip.nunique()
device_model_unique = df.device_model.nunique()
device_type_unique = df.device_type.nunique()
print(f'Unique value for site_id is {site_id_unique}')
print(f'Unique value for site_domain is {site_domain_unique}')
print(f'Unique value for site_category is {site_category_unique}')
print(f'Unique value for device_id is {device_id_unique}')
print(f'Unique value for device_ip is {device_ip_unique}')
print(f'Unique value for device_model is {device_model_unique}')
print(f'Unique value for device_type is {device_type_unique}')



# check the unknown feature C1 ....
print(df.C1.value_counts()/len(df))
c1_cat = df['C1'].unique()
check_cat = [print('c1 cat: {} and CTR: {}'.format(i, df.loc[np.where((df.C1 == i))].click.mean())) for i in c1_cat]
del check_cat
df.groupby(['C1', 'click']).size().unstack().plot(kind= 'bar', figsize = (12, 6), title = 'C1 total volume and click volumne')
### this tells although 1005 got the highest volume  in total, its CTR is not high as expected.

df_c1_count = df[['C1', 'click']].groupby(['C1']).count().reset_index()
df_c1_count = df_c1_count.rename(columns = {'click': 'impression'})
df_c1_count['click'] = df_click[['C1', 'click']].groupby(['C1']).count().reset_index()['click']
df_c1_count['CTR'] = df_c1_count['click'] / df_c1_count['impression']
plt.figure(figsize = (12, 6))
sns.barplot(y='CTR', x='C1', data=df_c1_count)
plt.title('distribution of click for C1')
### from thids graph, can clearly see 100 got the CTR higher than 20% roughly
### generally, we could say 1008 has the most ccontribution value among all the others with the data ratio of 0.01% contributing 18% of CTR.
### furthermore, also discover that 1002 contribute the second most as its data ratio and CTR are 5.6% and 21% (the lowest raito of data got the surprisingly high CTR)



print("unique value of C14 is {} ".format(df.C14.nunique()))
print("unique value of C15 is {} ".format(df.C15.nunique()))
print("unique value of C16 is {} ".format(df.C16.nunique()))
print("unique value of C17 is {} ".format(df.C17.nunique()))
print("unique value of C18 is {} ".format(df.C18.nunique()))
print("unique value of C19 is {} ".format(df.C19.nunique()))
print("unique value of C20 is {} ".format(df.C20.nunique()))
df.groupby(['C15', 'click']).size().unstack().plot(kind= 'bar', title= 'C15 distribution')
### as we can see from the graph above, we can just combine group 120 1024 480 768 together to a single group, 
### or tring to be conservative by cross-calculating the simiilarities between groups and assign low data group to corressponding one.

df.groupby(['C18', 'click']).size().unstack().plot(kind= 'bar', title= 'C15 distribution')
df.groupby(['C16', 'click']).size().unstack().plot(kind= 'bar', title= 'C15 distribution')
### same situation applied here

df.groupby(['C18', 'click']).size().unstack().plot(kind= 'bar', title= 'C15 distribution')
### for C18  no further combining needed
