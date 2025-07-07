path = r"C:\Users\HP\Downloads\data.csv"
df = pd.read_csv(path, dtype={0:'str'})
df.head(1)
df.info()
df[df.duplicated()]
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['month'] = df['date'].dt.month
df['converted'] = np.where(df['transactions'] > 0, 'converted', 'not_converted')
df['bounces'] = df['bounces'].fillna(0).astype(int)
df['pageviews'] = df['pageviews'].fillna(0).astype(int)
df['transactionRevenue'] = df['transactionRevenue'].fillna(0).astype(float)
df['transactions'] = df['transactions'].fillna(0).astype(float)
df['timeOnSite'] = df['timeOnSite'].fillna(0).astype(float)
df['timeOnSite_m'] = (df['timeOnSite'] / 60).astype(float)
c = df['converted'].value_counts()
plt.pie(c, labels=[f"{i}\n{v:,} ({v/c.sum()*100:.1f}%)" for i, v in c.items()],
        colors=['#aec7e8', '#1f77b4'], startangle=140, wedgeprops={'edgecolor': 'white'})
plt.title('Distribución de sesiones por conversión')
plt.axis('equal')
plt.show()
top3 = lambda s: s.value_counts().head(3).index.tolist()
df.groupby('converted').agg({col: 'median' for col in ['visitNumber', 'bounces', 'hits', 'pageviews', 'timeOnSite_m']} | {col: top3 for col in ['channelGrouping', 'browser', 'deviceCategory', 'country']}).reset_index()
df['hits_per_minute'] = df['hits'] / df['timeOnSite_m']
df['seconds_per_pageview'] = df['timeOnSite'] / df['pageviews']
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
user_converted = df.groupby('fullVisitorID')['transactions'].sum() > 0
df['has_ever_converted'] = df['fullVisitorID'].map(user_converted).astype(int)
visit_range = df.groupby('fullVisitorID')['date'].agg(['min', 'max'])
visit_range['days_between_first_last_visit'] = (visit_range['max'] - visit_range['min']).dt.days
df['days_between_first_last_visit'] = df['fullVisitorID'].map(visit_range['days_between_first_last_visit'])
df['is_first_visit'] = (df['visitNumber'] == 1).astype(int)
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
features = ['is_first_visit', 'bounces', 'hits_per_minute', 'seconds_per_pageview', 'is_weekend', 'has_ever_converted', 'days_between_first_last_visit']
X = df[features]
df['converted_n'] = (df['converted'] == 'converted').astype(int)
y = df['target']
preprocessor = ColumnTransformer([('num', StandardScaler(), X.columns)])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
pipe_lr = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
pipe_rf = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
cv_results = cross_validate(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='recall',
    return_train_score=True
)
param_grid = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
