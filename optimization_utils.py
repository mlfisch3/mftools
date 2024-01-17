from bayes_opt import BayesianOptimization 
from hyperopt import hp, fmin, tpe

#  gbm_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample): 
#  gbm_cl_bo2(params):

# source: https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/
# Gradient Boosting Machine 
def gbm_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample): 
    params_gbm = {} params_gbm['max_depth'] = round(max_depth)
    params_gbm['max_features'] = max_features 
    params_gbm['learning_rate'] = learning_rate
    params_gbm['n_estimators'] = round(n_estimators) 
    params_gbm['subsample'] = subsample 
    scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gbm), X_train, y_train, scoring=acc_score, cv=5).mean() 
    score = scores.mean() 

    return score # Run Bayesian Optimization 

start = time.time() 
params_gbm ={ 'max_depth':(3, 10), 'max_features':(0.8, 1), 'learning_rate':(0.01, 1), 'n_estimators':(80, 150), 'subsample': (0.8, 1) } 
gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111) 
gbm_bo.maximize(init_points=20, n_iter=4) 
print('It takes %s minutes' % ((time.time() - start)/60))


params_gbm = gbm_bo.max['params'] 
params_gbm['max_depth'] = round(params_gbm['max_depth'])
params_gbm['n_estimators'] = round(params_gbm['n_estimators']) 
params_gbm

    # OUTPUT:
    # {'learning_rate': 0.07864837617488214,
    #  'max_depth': 6,
    #  'max_features': 0.8723008386644597,
    #  'n_estimators': 113,
    #  'subsample': 0.8358969695415375}



# Run Bayesian Optimization from hyperopt
start = time.time()
space_lr = {'max_depth': hp.randint('max_depth', 3, 10),
            'max_features': hp.uniform('max_features', 0.8, 1),
            'learning_rate': hp.uniform('learning_rate',0.01, 1),
            'n_estimators': hp.randint('n_estimators', 80,150),
            'subsample': hp.uniform('subsample',0.8, 1)}

def gbm_cl_bo2(params):
    params = {'max_depth': params['max_depth'],
              'max_features': params['max_features'],
              'learning_rate': params['learning_rate'],
              'n_estimators': params['n_estimators'],
              'subsample': params['subsample']}
    gbm_bo2 = GradientBoostingClassifier(random_state=111, **params)
    best_score = cross_val_score(gbm_bo2, X_train, y_train, scoring=acc_score, cv=5).mean()
    return 1 - best_score

gbm_best_param = fmin(fn=gbm_cl_bo2,
                space=space_lr,
                max_evals=24,
                rstate=np.random.RandomState(42),
                algo=tpe.suggest)
print('It takes %s minutes' % ((time.time() - start)/60))

gbm_best_param

    #OUTPUT:
    # {'learning_rate': 0.03516615427790515,
    #  'max_depth': 6,
    #  'max_features': 0.8920776081423815,
    #  'n_estimators': 148,
    #  'subsample': 0.9981549036976672}
