import mysql.connector

team = 5

# Requirement1: create schema ( name: DMA_team## )
def requirement1(host, user, password):
    cnx = mysql.connector.connect(host=host,user=user,password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    # 컴파일 오류 등으로 database 및 table 생성이 중단 되었을 시, 
    # pk중복 오류를 방지하기 위해 이미 db가 있으면 drop을 시행.
    cursor.execute('DROP DATABASE IF EXISTS DMA_team%02d;' % team)
    # R2-1. DMA_team05 schema생성.
    # 이미 존재할 경우 중복 생성 방지를 위해 if not exist 조건
    cursor.execute('CREATE DATABASE IF NOT EXISTS DMA_team%02d;' % team)

    cursor.close()

    
# Requirement2: create table
def requirement2(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    # R2-1에서 create한 db사용 
    cursor.execute('USE DMA_team%02d;' % team)

    # R2-2. table 생성. 순서 일치. 
    # 중복 생성 방지를 위한 if not exist 조건.
    # foreign key 조건 작성 안함.

    # 1:create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users(
    user_id INT(11) NOT NULL,
    user_yelping_since_year INT(11) NOT NULL,      
    user_average_stars FLOAT(2,1) NOT NULL,
    user_votes_funny INT(11) NOT NULL,      
    user_votes_useful INT(11) NOT NULL,      
    user_votes_cool INT(11) NOT NULL,      
    user_fans INT(11) NOT NULL,      
    PRIMARY KEY (user_id)); 
    ''')

    #2:create elite table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS elite(
    user_id INT(11) NOT NULL,      
    year_id INT(11) NOT NULL,      
    PRIMARY KEY (user_id,year_id)); 
    ''')

    #3:create years table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS years(
    year_id INT(11) NOT NULL,      
    actual_year INT(11) NOT NULL UNIQUE,      
    PRIMARY KEY (year_id)); 
    ''')

    #4:create users_compliments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users_compliments(
    compliment_id INT(11) NOT NULL,
    user_id INT(11) NOT NULL,
    number_of_compliments VARCHAR(255) NOT NULL,
    PRIMARY KEY (compliment_id,user_id));
    ''')

    #5:create compliments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS compliments(
    compliment_id INT(11) NOT NULL,
    compliment_type VARCHAR(255) NOT NULL UNIQUE,
    PRIMARY KEY (compliment_id));
    ''')

    #6:create reviews table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews(
    review_id INT(11) NOT NULL,
    business_id INT(11) NOT NULL,
    user_id INT(11) NOT NULL,
    review_stars INT(11) NOT NULL,
    review_votes_funny VARCHAR(255) NOT NULL,
    review_votes_useful VARCHAR(255) NOT NULL,
    review_votes_cool VARCHAR(255) NOT NULL,
    review_length VARCHAR(255) NOT NULL,
    PRIMARY KEY (review_id));
    ''')

    #7:create tips table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tips(
    business_id INT(11) NOT NULL,    
    user_id INT(11) NOT NULL,
    likes INT(11) NOT NULL,
    tip_length VARCHAR(255) NOT NULL,
    PRIMARY KEY (business_id,user_id));
    ''')

    #8:create business table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business(
    business_id INT(11) NOT NULL,
    active TINYINT(1) NOT NULL DEFAULT 0,
    city VARCHAR(255),
    state VARCHAR(255),
    stars FLOAT(2,1) NOT NULL,
    review_count VARCHAR(255) NOT NULL,
    PRIMARY KEY (business_id));
    ''')
    
    #9:create business_categories table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business_categories(
    business_id INT(11) NOT NULL,
    category_id INT(11) NOT NULL,
    PRIMARY KEY (business_id,category_id));
    ''')

    #10:create categories table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories(
    category_id INT(11) NOT NULL,
    category_name VARCHAR(255) NOT NULL UNIQUE,
    PRIMARY KEY (category_id));
    ''')
    
    #11:create business_attributes table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business_attributes(
    attribute_id INT(11) NOT NULL,
    business_id INT(11) NOT NULL,
    attribute_value VARCHAR(255) NOT NULL,
    PRIMARY KEY (business_id,attribute_id));
    ''')

    #12:create attributes table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attributes(
    attribute_id INT(11) NOT NULL,
    attribute_name VARCHAR(255) NOT NULL UNIQUE,
    PRIMARY KEY (attribute_id));
    ''')

    #13:create business_hours table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS business_hours(
    business_id INT(11) NOT NULL,
    day_id INT(11) NOT NULL,
    opening_time VARCHAR(255) NOT NULL,
    closing_time VARCHAR(255) NOT NULL,
    PRIMARY KEY (business_id,day_id));
    ''')
    
    #14:create days table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS days(
    day_id INT(11) NOT NULL,
    day_of_week VARCHAR(255) NOT NULL UNIQUE,
    PRIMARY KEY (day_id));
    ''')

    cursor.close()


# Requirement3: insert data
def requirement3(host, user, password, directory):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor(buffered=True)
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    cursor.execute('USE DMA_team%02d;' % team)

    # table의 이름(csv파일의 이름) 저장
    table_list=['users','elite','years','users_compliments','compliments','reviews','tips','business','business_categories','categories','business_attributes','attributes','business_hours','days']
    # table이름 뒤에 .csv추가
    filepath_list=['{}.csv'.format(table) for table in table_list]
    # csv file의 경로를 입력할 filepath list생성
    filepath=[]
    # directory를 이용해서 데이터 셋의 파일 경로 지정
    for i in range(len(filepath_list)):
        filepath.append(directory+'/'+filepath_list[i])

    # 빈 파일인지 검사하기 위한 명령어 변수 설정.
    empty_check='SELECT COUNT(*) FROM {};'

    #1:input datasets in users table
    # 파일을 순서대로 입력 및 수기로 입력할 때 인덱스 실수를 방지하기 위한 변수 지정.  
    index=0
    # empty파일 검사 시행
    cursor.execute(empty_check.format(table_list[index])) 
    # csv파일 불러와서 데이터 입력하기. TA시간에 진행한 코드 참고. 인코딩은 utf-8로 진행.
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        # 첫째줄[0]은 속성의 이름이므로 건너뛰고 두번째줄[1]부터 readlines 
        for row in csv_data.readlines()[1:]:
            # 데이터 입력시 "'OO'"와 같이 따옴표 두개로 들어오는 경우가 있었음.
            # 따라서 TA 코드(split,stripe)에 replace항목 추가. 
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                # data가 없는 항목은 null(string)로 입력
                if data == '':
                    row[idx] = 'null'
                # 3번째 속성은 float타입
                if idx == 2 :
                    row[idx] = float(data)
                # 그 이외는 모두 int타입
                else:
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            # null(string)을 NULL로 입력
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    # 다른 테이블도 마찬가지 방식으로 진행.
    #2:input datasets in elite table 
    index=index+1 
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in [0,1] :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()
    
    #3:input datasets in years table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in [0,1] :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()
    
    #4:input datasets in users_compliments table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in [0,1] :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()
    
    #5:input datasets in compliments table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx ==0 :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #6:input datasets in reviews table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in range(4) :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #7:input datasets in tips table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in range(3) :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #8:input datasets in business table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx==0 :
                    row[idx] = int(data)
                if idx==1:
                    if(row[idx]=='true'):
                        row[idx]=True
                    elif(row[idx]=='false'):
                        row[idx]=False
                    else:
                        row[idx]=True
                if idx==4:
                    row[idx]=float(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #9:input datasets in business_categories table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in range(2) :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #10:input datasets in categories table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx ==0 :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #11:input datasets in business_attributes table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in range(2) :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #12:input datasets in attributes table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx ==0 :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #13:input datasets in business_hours table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx in range(2) :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    #14:input datasets in days table  
    index=index+1
    cursor.execute(empty_check.format(table_list[index])) 
    with open(filepath[index], 'r', encoding='utf-8') as csv_data:
        for row in csv_data.readlines()[1:]:
            row = row.replace('"','').strip().split(',')
            for idx, data in enumerate(row):
                if data == '':
                    row[idx] = 'null'
                if idx ==0 :
                    row[idx] = int(data)
            row = tuple(row)
            sql='INSERT INTO {} VALUES {};'.format(table_list[index],row)
            sql = sql.replace('\'null\'', 'null')
            cursor.execute(sql)
    cnx.commit()

    cursor.close()

# Requirement4: add constraint (foreign key)
def requirement4(host, user, password):
    cnx = mysql.connector.connect(host=host, user=user, password=password)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')

    cursor.execute('USE DMA_team%02d;' % team)

    cursor.execute('ALTER TABLE elite ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES users(user_id);')
    cursor.execute('ALTER TABLE elite ADD CONSTRAINT FOREIGN KEY (year_id) REFERENCES years(year_id);')
    cursor.execute('ALTER TABLE users_compliments ADD CONSTRAINT FOREIGN KEY (compliment_id) REFERENCES compliments(compliment_id);')
    cursor.execute('ALTER TABLE users_compliments ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES users(user_id);')
    cursor.execute('ALTER TABLE tips ADD CONSTRAINT FOREIGN KEY (business_id) REFERENCES business(business_id)')
    cursor.execute('ALTER TABLE tips ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES users(user_id)')
    cursor.execute('ALTER TABLE business_categories ADD CONSTRAINT FOREIGN KEY (business_id) REFERENCES business(business_id)')
    cursor.execute('ALTER TABLE business_categories ADD CONSTRAINT FOREIGN KEY (category_id) REFERENCES categories(category_id)')
    cursor.execute('ALTER TABLE business_attributes ADD CONSTRAINT FOREIGN KEY (business_id) REFERENCES business(business_id)')
    cursor.execute('ALTER TABLE business_attributes ADD CONSTRAINT FOREIGN KEY (attribute_id) REFERENCES attributes(attribute_id)')
    cursor.execute('ALTER TABLE business_hours ADD CONSTRAINT FOREIGN KEY (business_id) REFERENCES business(business_id)')
    cursor.execute('ALTER TABLE business_hours ADD CONSTRAINT FOREIGN KEY (day_id) REFERENCES days(day_id)')  
    cursor.execute('ALTER TABLE reviews ADD CONSTRAINT FOREIGN KEY (business_id) REFERENCES business(business_id);')
    cursor.execute('ALTER TABLE reviews ADD CONSTRAINT FOREIGN KEY (user_id) REFERENCES users(user_id);')
    cursor.close()

host = 'localhost'
user = 'root'
password = '#######'
directory = './dataset'

requirement1(host=host, user=user, password=password)
requirement2(host=host, user=user, password=password)
requirement3(host=host, user=user, password=password, directory=directory)
requirement4(host=host, user=user, password=password)