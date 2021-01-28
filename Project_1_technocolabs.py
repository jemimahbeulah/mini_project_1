#!/usr/bin/env python
# coding: utf-8

# <h2 align='center'> PROJECT 1 </h2>

# In[1]:


# importing essential libraries and modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier


# In[2]:


# load training set 
train = pd.read_csv("technocolabs training set.csv")
train.shape


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


train.describe(include=['O'])


# In[7]:


train.drop_duplicates(inplace=True)
train.dropna(inplace=True)
train.shape


# In[8]:


# changing the type of columns 
for colname in ['skip_1','skip_2','skip_3','not_skipped','hist_user_behavior_is_shuffle','premium']:
    train[colname] = train[colname].astype(int, copy=False)


# In[9]:


train['skip'] = train['not_skipped'].replace({ 0 : 1, 1 : 0 })


# In[10]:


train['skip'].value_counts()


# In[11]:


train['skip'].value_counts().plot(kind='pie', autopct = "%1.0f%%")


# In[12]:


### Analysing categorical data


# In[13]:


col = ['skip_1','skip_2','skip_3',
       'not_skipped','context_switch','no_pause_before_play',
       'short_pause_before_play','long_pause_before_play','hist_user_behavior_is_shuffle',
       'premium','context_type','hist_user_behavior_reason_start',
       'hist_user_behavior_reason_end']

plt.figure(figsize=(20,25))
n = 1
for colname in col:
    plt.subplot(5,3,n)
    train[colname].value_counts().plot(kind='bar')
    plt.xlabel(colname)
    n +=1


# In[14]:


# creating copy of the train data 
df = train.copy()
df.shape


# In[15]:


#df.date = df.date.apply(pd.to_datetime)
#df.info()


# In[16]:


### Dropping Irrelevent columns


# In[17]:


df = df.drop(columns=['skip_1','skip_2','skip_3','not_skipped','date'])
df.shape


# In[18]:


df.head()


# In[19]:


### One-hot Encoding on Train data


# In[20]:


df1 = df.drop(['session_id', 'track_id_clean'], axis=1)
df1.shape


# In[21]:


dummy_train = pd.get_dummies(df1)
dummy_train.shape


# In[22]:


dummy_train.head()


# In[23]:


## Analysing Track Features 


# In[24]:


track = pd.read_csv("track_feats.csv")
track.head()


# In[25]:


track.shape


# In[26]:


track.info()


# In[27]:


track.duplicated().sum()


# In[28]:


track.isna().sum()


# In[29]:


### Statistical summary of track features


# In[30]:


track.describe()


# In[31]:


# summary for object columns:
track.describe(include='O')


# In[35]:


### EDA 


# In[42]:


# extracting columns other than float ie, int and object for eda :
track[[c for c in track.columns if track[c].dtype != 'float64']].head()


# In[43]:


# distribution of release years : 
sns.distplot(track.release_year)
plt.title("Distribution of Release Years");


# In[44]:


track['key'].unique()


# In[45]:


# we have unique values in keys columns: we have specific Pitch class for these keys(just naming keys acc. to their pitch class)
keys = track.key.value_counts().sort_index()
sns.barplot(
    x=[ "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    y=keys.values/keys.sum()
)
plt.title("Distribution of Track Keys")
plt.xlabel("Key");


# In[46]:


sns.countplot(track.time_signature)


# In[47]:


### Analysing numerical columns


# In[48]:


track.hist(figsize=(20,15));


# In[50]:


### Correlation matrix  


# In[52]:


plt.figure(figsize=(20,15))
sns.heatmap(track.corr(), annot=True);


# In[53]:


### Merging track and Training Data on the basis of track_id (primary key):


# In[54]:


track.shape


# In[55]:


df.shape


# In[56]:


df.rename(columns={'track_id_clean': 'track_id'}, inplace=True)


# In[57]:


final_train = pd.merge(df, track, on=['track_id'], left_index=True, right_index=False, sort=True)
final_train.shape


# In[58]:


final_train.sort_values(axis=0, by=['session_id','session_position'], inplace=True)
final_train.reset_index(drop=True,inplace=True)


# In[59]:


final_train.head()


# In[60]:


ft = final_train.drop(columns=["session_id","track_id"])
ft = pd.get_dummies(ft, drop_first=True)
ft.shape


# In[61]:


ft.info()


# In[62]:


### Modeling with only Training Data 


# In[63]:


dummy_train.head(2)


# In[64]:


dummy_train.shape


# In[65]:


### Train Test Split 


# In[66]:


X = dummy_train.drop(columns=["skip"])
y = dummy_train.skip
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=2
)


# In[67]:


##### Applying LogisticRegression


# In[68]:


from sklearn.linear_model import LogisticRegression

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# In[69]:


### Standardsing Data


# In[70]:


scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)


# Applying LOgistic Regression
log = LogisticRegressionCV(
    cv=3
).fit(
    sX_train,
    y_train
)

print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))


# In[71]:


###  Applying Random Forest


# In[72]:


rfc = RandomForestClassifier(
    n_estimators=100
).fit(
    X_train,
    y_train
)

print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))


# In[73]:


### Feature selection using Boruta 


# In[74]:


get_ipython().system('pip install boruta')
from boruta import BorutaPy


# In[75]:


rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)   # initialize the boruta selector
boruta_selector.fit(np.array(sX_train), np.array(y_train))  


# In[76]:


print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)


# In[77]:


selected_rf_features = pd.DataFrame({'Feature':list(X_train.columns),
                                      'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')


# In[78]:


X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))


# In[79]:


### Creating model with important features (selected by boruta)


# In[80]:


rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)


# In[81]:


print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test)) 


# In[82]:


### Applying XG Boost


# In[83]:


xg = xgb.XGBClassifier()
xg.fit(X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)


# In[84]:


print("XGB Train score: %s" % xg.score(X_important_train,y_train))
print("XGB Val score:   %s" % xg.score(X_important_val,y_val))
print("XGB Test score:  %s" % xg.score(X_important_test,y_test))


# In[85]:


### Testing  Model on the Data (given by technocolabs) 


# In[87]:


# loading test data 
ts1 = pd.read_csv('test_data.csv')
ts2 = pd.read_csv('test_data_20.csv')

test_set = pd.concat([ts1,ts2])
test_set.shape


# In[88]:


test_set.head(2)


# In[89]:


test_set['skip'] =  test_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_test_data = test_set.skip


# In[90]:


t1 = test_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean','hist_user_behavior_reason_end_appload'], 
              axis=1)
t1.shape


# In[91]:


#### Loading Validation data


# In[93]:


vs1 = pd.read_csv('val_data.csv')
vs2 = pd.read_csv('val_data_20.csv')

val_set = pd.concat([vs1,vs2])
val_set.shape


# In[94]:


val_set['skip'] =  val_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_val_data = val_set.skip


# In[95]:


v1 = val_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean', 'hist_user_behavior_reason_end_appload'],
             axis=1)
v1.shape


# In[96]:


### Selecting relevent features from Test and Validation set by using Boruta 


# In[97]:


X_val_set = boruta_selector.transform(np.array(v1))
X_test_set = boruta_selector.transform(np.array(t1))


# In[98]:


### Validation and test score on unseen data


# In[99]:


print("XGB Val score:   %s" % xg.score(X_val_set, y_val_data))
print("XGB Test score:  %s" % xg.score(X_test_set, y_test_data))


# In[100]:


print("RF Val score:   %s" % rf_important.score(X_val_set, y_val_data))
print("RF Test score:  %s" % rf_important.score(X_test_set, y_test_data))


# In[101]:


### Creating Model with final data (merged data-- ie track + train data)


# In[102]:


ft.head(2)


# In[103]:


X = ft.drop(columns=["skip"])
y = ft.skip
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=2
)


# In[104]:


### Scaling Data


# In[105]:


scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)


# Applying Logistic Regression
log = LogisticRegressionCV(
    cv=3
).fit(
    sX_train,
    y_train
)

print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))


# In[106]:


### Applying Random forest Classifier


# In[107]:


rfc = RandomForestClassifier(
    n_estimators=100
).fit(
    X_train,
    y_train
)

print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))


# In[108]:


### Feature selction using Boruta


# In[109]:


rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)   # initialize the boruta selector
boruta_selector.fit(np.array(sX_train), np.array(y_train))  


# In[110]:


print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)


# In[111]:


selected_rf_features = pd.DataFrame({'Feature':list(X_train.columns),
                                      'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')


# In[112]:


X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))


# In[113]:


rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)


# In[114]:


print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test)) 


# In[115]:


xg = xgb.XGBClassifier()
xg.fit(X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)


# In[116]:


print("XGB Train score: %s" % xg.score(X_important_train,y_train))
print("XGB Val score:   %s" % xg.score(X_important_val,y_val))
print("XGB Test score:  %s" % xg.score(X_important_test,y_test)) 


# In[117]:


##LGBM


# In[118]:


lgbm = LGBMClassifier( ).fit( X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)

print()
print("LGBM Train score: %s" % lgbm.score(X_important_train,y_train))
print("LGBM Val score:   %s" % lgbm.score(X_important_val,y_val))
print("LGBM Test score:  %s" % lgbm.score(X_important_test,y_test))


# <h2 align='center'> THANK YOU </h2>

# In[ ]:




