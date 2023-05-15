# IMPORT LIBRARIES NEEDED FOR PROJECT 2

import mysql.connector
import os
import pickle

import surprise
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
import graphviz


from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree

from mlxtend.frequent_patterns import association_rules, apriori

from surprise import SVD
from surprise import SVDpp
from surprise import NMF

np.random.seed(0)

# CHANGE GRAPHVIZ DIRECTORY
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'    #변경필요
#os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.47.1/bin/' # for MacOS

# CHANGE MYSQL INFORMATION, team number 
HOST = 'localhost'
USER = 'root'
PASSWORD = '111111'
SCHEMA = 'dma_team03'
team = 3

debug = 'on'

# PART 1: Decision tree 
def part1():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor(buffered = True)
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # Requirement 1-1. MAKE vip column
    
    cursor.execute(
       '''ALTER TABLE users ADD vip TINYINT(1) default 0;''')


    filepath = './dataset/vip_list.csv'
    with open(filepath, 'r', encoding='utf-8') as csv_data:
        next(csv_data, None)  # skip the headers
        row_count = 0
        for row in csv_data:
            # Change the null data
            row = row.strip().split(',')
            for i in range(len(row)):
                row[i] = row[i].replace('"', '')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = None
            row = tuple(row)
            cursor.execute(f'UPDATE users SET vip = 1 WHERE user_id = {row[0]};')
        cnx.commit()



    # Requirement 1-2. WRITE MYSQL QUERY AND EXECUTE. SAVE to .csv file

    cursor.execute('''
    SELECT users.user_id,
    (SELECT users.vip) AS vip_list,
    (SELECT users.user_yelping_since_year) AS user_yelping_since_year,
    (SELECT COUNT(*) FROM reviews AS r WHERE users.user_id = r.user_id) AS user_review_counts, 
    (SELECT users.user_fans) AS user_fans,
    (SELECT users.user_votes_funny) AS user_votes_funny,
    (SELECT users.user_votes_useful) AS user_votes_useful, 
    (SELECT users.user_votes_cool) AS user_votes_cool,
    (SELECT users.user_average_stars) AS user_average_stars,
    (SELECT SUM(likes) FROM tips AS t WHERE users.user_id = t.user_id) AS user_tip_counts
     FROM users;''')


    df1 = pd.DataFrame(cursor.fetchall())
    df1.columns = cursor.column_names
    df1.set_index('user_id', inplace = True)
    df1.fillna(0, inplace= True)
    df1.to_csv('DMA_project2_team%02d_part1.csv' % team)







    # Requirement 1-3. MAKE AND SAVE DECISION TREE
    # gini file name: DMA_project2_team##_part1_gini.pdf
    # entropy file name: DMA_project2_team##_part1_entropy.pdf

    features_col = [i for i in df1.columns if i != 'vip_list']
    features = df1[features_col]

    #Decision Tree(gini)
    DT_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=8, max_depth=4)
    X = df1[features_col]
    Y = df1['vip_list']
    DT_gini.fit(X,Y)

    #Visualization Decision Tree(gini)
    graph = tree.export_graphviz(DT_gini, out_file=None, feature_names=features_col, class_names=['normal', 'BEST'])
    graph = graphviz.Source(graph)
    graph.render('DMA_project2_team03_part1_gini', view=False)

    #Decision Tree(entropy)
    DT_entropy = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=8, max_depth=4)

    #visualization Decision Tree(entropy)
    DT_entropy.fit(X, Y)

    graph = tree.export_graphviz(DT_entropy, out_file=None, feature_names=features_col, class_names=['normal', 'BEST'])
    graph = graphviz.Source(graph)
    graph.render('DMA_project2_team03_part1_entropy', view=False)

    # Requirement 1-4. Don't need to append code for 1-4

    # new model 1, features 제거

    #check feature_importances
    print("\nDT_gini의 feature_importances는? \n")
    for i in range(0,8):
        print('{0} : {1}'.format(X.columns[i], DT_gini.feature_importances_[i]))

    print("\nDT_entropy의 feature_importances는? \n")
    for i in range(0, 8):
        print('{0} : {1}'.format(X.columns[i], DT_entropy.feature_importances_[i]))
    print("\n")
    # importance가 가장 낮은 세 가지 변수를 제거

    features_col_new = [i for i in df1.columns if (
                    (i != 'vip_list') and (i != 'user_votes_funny') and (i != 'user_tip_counts') and (
                        i != 'user_review_counts'))]
    features_new = df1[features_col_new]

    #Decision Tree(gini)
    DT_new_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=8, max_depth=4)
    X_new = df1[features_col_new]
    Y_new = df1['vip_list']
    DT_new_gini.fit(X_new,Y_new)

    # new model 2 GridSearchCV를 이용한 best parameter 찾기
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score


    param = {'criterion' : ['gini', 'entropy'], 'min_samples_leaf' : [6,8,10,12], 'max_depth' : [3,4,5,6]}
    model = tree.DecisionTreeClassifier()
    grid_dtree = GridSearchCV(model, param_grid=param, cv=5, refit=True, return_train_score=True)
    grid_dtree.fit(X, Y)
    print("\nnew model2에서 최고로 선정된 parameters : {0}\n".format(grid_dtree.best_params_))

    # 교차검증을 이용해 평가

    # origin
    score = cross_val_score(DT_gini, X, Y, scoring='accuracy', cv=5)
    print(f"origin gini model 교차검증 평균 점수 : {np.mean(score)}")

    # new 1
    score = cross_val_score(DT_new_gini, X_new, Y_new, scoring='accuracy', cv=5)
    print(f"new model 1 (feature 3개 제거) 교차검증 평균 점수 : {np.mean(score)}")

    # new 2
    new_DT = grid_dtree.best_estimator_
    score = cross_val_score(new_DT, X, Y,scoring='accuracy', cv=5)
    print(f"new model 2 (samples_leaf와 depth 수정) 교차검증 평균 점수 : {np.mean(score)}")




    cursor.close()
    

# PART 2: Association analysis
def part2():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)

    # Requirement 2-1. CREATE VIEW AND SAVE to .csv file
    fopen = open('DMA_project2_team%02d_part2_category.csv' % team, 'w', encoding='utf-8')
    
    cursor.execute('''
    CREATE OR REPLACE VIEW category_score AS 
    SELECT c.category_id as category_id, c.category_name as category_name, num_business, num_reviews, ctg_avg_stars, ctg_avg_stars*(num_reviews/num_business) AS score
    FROM categories c,
    (SELECT category_id, COUNT(*) AS num_business FROM business_categories GROUP BY category_id) AS bnum,
    (SELECT bc.category_id, SUM(tnr) AS num_reviews, SUM(sr)/SUM(tnr) AS ctg_avg_stars FROM business_categories bc,(SELECT business_id, COUNT(review_id) AS tnr,SUM(review_stars) AS sr FROM reviews GROUP BY business_id) AS review WHERE bc.business_id=review.business_id GROUP BY category_id) AS r
    WHERE c.category_id=bnum.category_id AND c.category_id=r.category_id
    ORDER BY score DESC
    LIMIT 30;
    ''')
    cursor.execute('SELECT * FROM category_score')
    df1=pd.DataFrame(cursor.fetchall())
    df1.columns=cursor.column_names
    df1.to_csv('DMA_project2_team%02d_part2_category.csv' % team)
    fopen.close()


    # Requirement 2-2. CREATE 2 VIEWS AND SAVE partial one to .csv file
    # User category rating view
    cursor.execute('''
        CREATE OR REPLACE VIEW user_category_rating AS
        SELECT f.user_id, f.category_name, (LEAST(f.num,5)+2*IFNULL(s.onum,0)) AS rating
        FROM (SELECT r.user_id, cs.category_id, cs.category_name, COUNT(cs.category_id) AS num FROM reviews r,category_score cs, business_categories bc WHERE r.business_id=bc.business_id AND bc.category_id=cs.category_id GROUP BY r.user_id, category_id ORDER BY user_id,category_id) AS f
        LEFT JOIN
        (SELECT r.user_id, cs.category_id, COUNT(cs.category_id) AS onum FROM reviews r,category_score cs, business_categories bc WHERE r.business_id=bc.business_id AND bc.category_id=cs.category_id AND r.review_stars>=4 GROUP BY r.user_id, category_id ORDER BY user_id,category_id) AS s
        ON f.user_id=s.user_id AND f.category_id=s.category_id
    ''')
    # Partial user category rating view
    fopen = open('DMA_project2_team%02d_part2_UCR.csv' % team, 'w', encoding='utf-8')
    cursor.execute('''
        CREATE OR REPLACE VIEW partial_user_category_rating AS
        SELECT ucr.user_id AS user, ucr.category_name AS category, ucr.rating 
        FROM user_category_rating ucr
        RIGHT JOIN
        (SELECT a.user_id, a.rating_num FROM (SELECT user_id, COUNT(*) AS rating_num 
        FROM user_category_rating GROUP BY user_id)a WHERE rating_num >= 10) AS rn
        ON ucr.user_id=rn.user_id
    ''')
    cursor.execute("SELECT * FROM partial_user_category_rating")
    df2=pd.DataFrame(cursor.fetchall())
    df2.columns=cursor.column_names
    partial_ucr=df2.set_index("user")
    partial_ucr.to_csv('DMA_project2_team%02d_part2_UCR.csv' % team)
    fopen.close()

    # Requirement 2-3. MAKE HORIZONTAL VIEW
    # file name: DMA_project2_team##_part2_horizontal.pkl
    # use to_pickle(): df.to_pickle(filename)
    ucr=pd.read_csv('DMA_project2_team%02d_part2_UCR.csv' % team)
    category_set=set(ucr.category.values)

    lquery=[]
    for category in category_set:
        query='MAX(IF(category="{}",1,0)) AS "{}"'.format(category,category)
        lquery.append(query)
    jquery=','.join(lquery)

    cursor.execute('''
    SELECT user,{}
    FROM partial_user_category_rating
    GROUP BY user
    '''.format(jquery))

    df3=pd.DataFrame(cursor.fetchall())
    df3.columns=cursor.column_names
    df3=df3.set_index('user')
    df3.to_pickle('DMA_project2_team%02d_part2_horizontal.pkl' % team)
    # print output
    print('Horizontal View')
    with open('DMA_project2_team%02d_part2_horizontal.pkl' % team,'rb') as files:
        print(pickle.load(files))


    # Requirement 2-4. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part2_association.pkl (pandas dataframe)
    frequent_itemset=apriori(df3,min_support=0.15,use_colnames=True)
    # print output
    print('Support')
    print(frequent_itemset)
    
    rules=association_rules(frequent_itemset,metric='lift',min_threshold=3)
    # write a pickle file
    rules.to_pickle('DMA_project2_team%02d_part2_association.pkl' % team)
    # print output
    print('Association')
    with open('DMA_project2_team%02d_part2_association.pkl' % team,'rb') as afiles:
        print(pickle.load(afiles))
    # write a csv file
    rules.to_csv('DMA_project2_team%02d_part2_association.csv' % team)

    cursor.close()


# Requirement 3-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n, user_based=True):
    results = defaultdict(list)
    if user_based:
        # testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장
        # Hint: testset은 (user_id, category_name, rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for user_id in testset:
            if user_id[0] in id_list:
                testset_id.append(user_id)
        predictions = algo.test(testset_id)
        for uid, cname, true_r, est, _ in predictions:
            # results는 user_id를 key로, [(category_name, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[uid].append((cname, est))
            pass
    else:
        # testset의 데이터 중 category name이 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, category_name, rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for category_name in testset:
            if category_name[1] in id_list:
                testset_id.append(category_name)
        predictions = algo.test(testset_id)
        for uid, cname, true_r, est, _ in predictions:
            # results는 category_name를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[cname].append((uid, est))
            pass
    for id_, ratings in results.items():
        # rating 순서대로 정렬하고 top-n개만 유지
        ratings = sorted(ratings, key = lambda x : x[-1], reverse = True)
        ratings = ratings[:n]
        results[id_] = ratings
        pass

    return results


# Requirement 3-1. WRITE get_top_n
def get_top_n(algo, testset, id_list, n, user_based=True):
    results = defaultdict(list)
    if user_based:
        # testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장
        # Hint: testset은 (user_id, category_name, rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for user_id in testset:
            if user_id[0] in id_list:
                testset_id.append(user_id)
        predictions = algo.test(testset_id)
        for uid, cname, true_r, est, _ in predictions:
            # results는 user_id를 key로, [(category_name, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[uid].append((cname, est))
            pass
    else:
        # testset의 데이터 중 category name이 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, category_name, rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for category_name in testset:
            if category_name[1] in id_list:
                testset_id.append(category_name)
        predictions = algo.test(testset_id)
        for uid, cname, true_r, est, _ in predictions:
            # results는 category_name를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[cname].append((uid, est))
            pass
    for id_, ratings in results.items():
        # rating 순서대로 정렬하고 top-n개만 유지
        ratings = sorted(ratings, key = lambda x : x[-1], reverse = True)
        ratings = ratings[:n]
        results[id_] = ratings
        pass

    return results


# PART 3. Requirement 3-2, 3-3, 3-4
def part3():
    file_path = 'DMA_project2_team%02d_part2_UCR.csv' % team
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 10), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset1 = data.build_full_trainset()
    testset1 = trainset1.build_anti_testset()
    
    trainset2 = data.build_full_trainset()
    testset2 = trainset2.build_anti_testset()
    
    trainset3 = data.build_full_trainset()
    testset3 = trainset3.build_anti_testset()

    # Requirement 3-2. User-based Recommendation
    sim_options_UC = {'name' : 'cosine', 'user_based' : True}
    sim_options_UP = {'name' : 'pearson', 'user_based' : True}
    sim_options_UM = {'name' : 'msd', 'user_based' : True}
    sim_options_UPb = {'name' : 'pearson_baseline', 'user_based' : True}

    algo_UBC = surprise.KNNBasic(sim_options = sim_options_UC)
    algo_UMP = surprise.KNNWithMeans(sim_options = sim_options_UP)

    algo_UBP = surprise.KNNBasic(sim_options = sim_options_UP)
    algo_UBM = surprise.KNNBasic(sim_options = sim_options_UM)
    algo_UBPb = surprise.KNNBasic(sim_options = sim_options_UPb)

    algo_UMC = surprise.KNNWithMeans(sim_options = sim_options_UC)
    algo_UMM = surprise.KNNWithMeans(sim_options = sim_options_UM)
    algo_UMPb = surprise.KNNWithMeans(sim_options = sim_options_UPb)

    algo_UBlC = surprise.KNNBaseline(sim_options = sim_options_UC)
    algo_UBlP = surprise.KNNBaseline(sim_options = sim_options_UP)
    algo_UBlM = surprise.KNNBaseline(sim_options = sim_options_UM)
    algo_UBlPb = surprise.KNNBaseline(sim_options = sim_options_UPb)

    algo_UZC = surprise.KNNWithZScore(sim_options = sim_options_UC)
    algo_UZP = surprise.KNNWithZScore(sim_options = sim_options_UP)
    algo_UZM = surprise.KNNWithZScore(sim_options = sim_options_UM)
    algo_UZPb = surprise.KNNWithZScore(sim_options = sim_options_UPb)

    uid_list = ['20384', '33306', '46833', '70628', '535']

    # set algorithm for 3-2-1
    algo = algo_UBC
    algo.fit(trainset1)
    results = get_top_n(algo, testset1, uid_list, n=5, user_based=True)
    with open('3-2-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # set algorithm for 3-2-2
    algo = algo_UMP
    algo.fit(trainset1)
    results = get_top_n(algo, testset1, uid_list, n=5, user_based=True)
    with open('3-2-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # 3-2-3. Best Model
    UB = {}

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBC.fit(trainset1)
        predictions = algo_UBC.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBasic, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBP.fit(trainset1)
        predictions = algo_UBP.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBasic, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBM.fit(trainset1)
        predictions = algo_UBM.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBasic, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBPb.fit(trainset1)
        predictions = algo_UBPb.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBasic, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UMC.fit(trainset1)
        predictions = algo_UMC.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithMeans, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UMP.fit(trainset1)
        predictions = algo_UMP.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithMeans, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UMM.fit(trainset1)
        predictions = algo_UMM.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithMeans, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UMPb.fit(trainset1)
        predictions = algo_UMPb.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithMeans, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBlC.fit(trainset1)
        predictions = algo_UBlC.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBaseline, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBlP.fit(trainset1)
        predictions = algo_UBlP.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBaseline, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBlM.fit(trainset1)
        predictions = algo_UBlM.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBaseline, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UBlPb.fit(trainset1)
        predictions = algo_UBlPb.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNBaseline, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UZC.fit(trainset1)
        predictions = algo_UZC.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithZScore, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UZP.fit(trainset1)
        predictions = algo_UZP.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithZScore, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UZM.fit(trainset1)
        predictions = algo_UZM.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithZScore, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset1, testset1) in enumerate(kf.split(data)):
        algo_UZPb.fit(trainset1)
        predictions = algo_UZPb.test(testset1)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    UB['KNNWithZScore, Pearson_baseline'] = np.mean(acc)

    print(UB, '\n')
    best_algo_ub = min(UB, key=UB.get)
    print('The best model for user-based recommendation is', best_algo_ub + '.\n')


    # Requirement 3-3. Item-based Recommendation
    sim_options_IC = {'name' : 'cosine', 'user_based' : False}
    sim_options_IP = {'name' : 'pearson', 'user_based' : False}
    sim_options_IM = {'name' : 'msd', 'user_based' : False}
    sim_options_IPb = {'name' : 'pearson_baseline', 'user_based' : False}

    algo_IBC = surprise.KNNBasic(sim_options = sim_options_IC)
    algo_IMP = surprise.KNNWithMeans(sim_options = sim_options_IP)

    algo_IBP = surprise.KNNBasic(sim_options = sim_options_IP)
    algo_IBM = surprise.KNNBasic(sim_options = sim_options_IM)
    algo_IBPb = surprise.KNNBasic(sim_options = sim_options_IPb)

    algo_IMC = surprise.KNNWithMeans(sim_options = sim_options_IC)
    algo_IMM = surprise.KNNWithMeans(sim_options = sim_options_IM)
    algo_IMPb = surprise.KNNWithMeans(sim_options = sim_options_IPb)

    algo_IBlC = surprise.KNNBaseline(sim_options = sim_options_IC)
    algo_IBlP = surprise.KNNBaseline(sim_options = sim_options_IP)
    algo_IBlM = surprise.KNNBaseline(sim_options = sim_options_IM)
    algo_IBlPb = surprise.KNNBaseline(sim_options = sim_options_IPb)

    algo_IZC = surprise.KNNWithZScore(sim_options = sim_options_IC)
    algo_IZP = surprise.KNNWithZScore(sim_options = sim_options_IP)
    algo_IZM = surprise.KNNWithZScore(sim_options = sim_options_IM)
    algo_IZPb = surprise.KNNWithZScore(sim_options = sim_options_IPb)

    cname_list = ['Irish',
                'Ethiopian',
                'Wine Bars',
                'Vegetarian',
                'Sushi Bars']

    # set algorithm for 3-3-1
    algo = algo_IBC
    algo.fit(trainset2)
    results = get_top_n(algo, testset2, cname_list, n=10, user_based=False)
    with open('3-3-1.txt', 'w') as f:
        for cname, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Category NAME %s top-10 results\n' % cname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # set algorithm for 3-3-2
    algo = algo_IMP
    algo.fit(trainset2)
    results = get_top_n(algo, testset2, cname_list, n=10, user_based=False)
    with open('3-3-2.txt', 'w') as f:
        for cname, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Category NAME %s top-10 results\n' % cname)
            for uid, score in ratings:
                f.write('User ID %s\n\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # 3-3-3. Best Model
    IB = {}

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBC.fit(trainset2)
        predictions = algo_IBC.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBasic, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBP.fit(trainset2)
        predictions = algo_IBP.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBasic, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBM.fit(trainset2)
        predictions = algo_IBM.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBasic, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBPb.fit(trainset2)
        predictions = algo_IBPb.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBasic, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IMC.fit(trainset2)
        predictions = algo_IMC.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithMeans, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IMP.fit(trainset2)
        predictions = algo_IMP.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithMeans, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IMM.fit(trainset2)
        predictions = algo_IMM.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithMeans, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IMPb.fit(trainset2)
        predictions = algo_IMPb.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithMeans, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBlC.fit(trainset2)
        predictions = algo_IBlC.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBaseline, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBlP.fit(trainset2)
        predictions = algo_IBlP.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBaseline, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBlM.fit(trainset2)
        predictions = algo_IBlM.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBaseline, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IBlPb.fit(trainset2)
        predictions = algo_IBlPb.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNBaseline, Pearson_baseline'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IZC.fit(trainset2)
        predictions = algo_IZC.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithZScore, Cosine'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IZP.fit(trainset2)
        predictions = algo_IZP.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithZScore, Pearson'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IZM.fit(trainset2)
        predictions = algo_IZM.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithZScore, MSD'] = np.mean(acc)

    np.random.seed(0)
    kf = KFold(n_splits=5)
    acc=[]
    for i, (trainset2, testset2) in enumerate(kf.split(data)):
        algo_IZPb.fit(trainset2)
        predictions = algo_IZPb.test(testset2)
        acc.append(surprise.accuracy.rmse(predictions, verbose=True))
    IB['KNNWithZScore, Pearson_baseline'] = np.mean(acc)

    print(IB, '\n')
    best_algo_ib = min(IB, key=IB.get)
    print('The best model for item-based recommendation is', best_algo_ib + '.\n')

    # Requirement 3-4. Matrix-factorization Recommendation   
    algo1 = surprise.SVD(n_factors=100, n_epochs=20, biased=False)
    algo2 = surprise.SVD(n_factors=200, n_epochs=20, biased=True)
    algo3 = surprise.SVDpp(n_factors=100, n_epochs=20)
    algo4 = surprise.SVDpp(n_factors=200, n_epochs=20)


    # set algorithm for 3-4-1
    algo = algo1
    algo.fit(trainset3)
    results = get_top_n(algo, testset3, uid_list, n=5, user_based=True)
    with open('3-4-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # set algorithm for 3-4-2
    algo = algo2
    algo.fit(trainset3)
    results = get_top_n(algo, testset3, uid_list, n=5, user_based=True)
    with open('3-4-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # set algorithm for 3-4-3
    algo = algo3
    algo.fit(trainset3)
    results = get_top_n(algo, testset3, uid_list, n=5, user_based=True)
    with open('3-4-3.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # set algorithm for 3-4-4
    algo = algo4
    algo.fit(trainset3)
    results = get_top_n(algo, testset3, uid_list, n=5, user_based=True)
    with open('3-4-4.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for cname, score in ratings:
                f.write('Category NAME %s\n\tscore %s\n' % (cname, str(score)))
            f.write('\n')

    # 3-4-5. Best Model
    svd_distributions = {'n_epochs' : range(100), 'n_factors' : range(200), 'biased' : ['False', 'True']}
    svdRs = surprise.model_selection.search.RandomizedSearchCV(SVD, svd_distributions, random_state=0)
    svdRs.fit(data)

    print('SVD')
    print(svdRs.best_score['rmse'])
    print(svdRs.best_params['rmse'], '\n')

    svdpp_distributions = {'n_epochs' : range(100), 'n_factors' : range(200)}
    svdppRs = surprise.model_selection.search.RandomizedSearchCV(SVDpp, svdpp_distributions, random_state=0)
    svdppRs.fit(data)

    print('SVD++')
    print(svdppRs.best_score['rmse'])
    print(svdppRs.best_params['rmse'], '\n')

    nmf_distributions = {'n_epochs' : range(100), 'n_factors' : range(200), 'biased' : ['False', 'True']}
    nmfRs = surprise.model_selection.search.RandomizedSearchCV(NMF, nmf_distributions, random_state=0)
    nmfRs.fit(data)

    print('NMF')
    print(nmfRs.best_score['rmse'])
    print(nmfRs.best_params['rmse'], '\n')

    

if __name__ == '__main__':
    
    part1()

    part2()
    part3()




