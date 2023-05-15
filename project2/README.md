# Project #2 Info
## DB mining & Automated Recommendation System
This project aims to construct a database using the method of Project #1, based on the additional data provided. It further aims to implement DB mining and an Automated Recommendation System using this database. The project is divided into three main parts.

**PART I** - *Decision Tree*  
**PART II** - *Association Analysis*  
**PART III** - *Recommendation System*  

### <ins>PART I. Decision Tree</ins>
PART I aims to create a decision tree based on the criteria for selecting VIP users from website A. In R1-1 and R1-2, the goal is to execute queries using mysql-connector-python to generate the data for constructing the decision tree. In R1-3, the objective is to create the decision tree using the sklearn library in Python and visualize the results using the graphviz library. In R1-4, the goal is to create a decision tree for determining VIP status that is different from the decision tree generated in R1-3. website A's database should satisfy the following requirements:

> **_(R1-1)_** Website A considers customers who contribute to the long-term maintenance and activation of the website as VIP users. The goal is to understand the criteria for selecting VIP users on this site. To begin with, in order to incorporate the provided vip_list.csv, a new column called "vip" needs to be added to the "users" table in the DMA_project2 database. The data type for "vip" should be TINYINT(1), and its default value should be 0. If an item's ID is included in vip_list.csv, the corresponding item's "vip" value should be set to 1.

> **_(R1-2)_** Write a query that returns the following columns. However, it should be written as a nested query in a single SQL statement, and the results should be saved in DMA_project2_team##_part1.csv.
> - user_id: ID of the customer
> - vip: Indicates whether the customer is a VIP
> - user_yelping_since_year: The year when the customer started using Site A
> - user_review_counts: The number of reviews written by the customer
> - user_fans: The number of fans the customer has
> - user_votes_funny: The number of funny votes the customer has received
> - user_votes_useful: The number of useful votes the customer has received
> - user_votes_cool: The number of cool votes the customer has received
> - user_average_stars: The average rating given by the customer
> - user_tip_counts: The sum of likes on tips written by the customer

> **_(R1-3)_** Create decision tree for the VIP selection criteria using the results obtained in R1-2. Create decision trees using two node impurity measurement methods: gini and entropy. The results should be saved using graphviz. The decision tree created in this problem should have the following attributes. The report should also include a brief analysis based on the decision tree created.
> - Used Library: sklearn.tree.DecisionTreeClassifier
> - Node impurity criterion: gini / entropy
> - Result filenames: Differentiate the files based on the node impurity criterion as DMA_project2_team##_part1_gini.pdf and DMA_project2_team##_part1_entropy.pdf.
> - Analysis goal: VIP selection criteria
> - min_samples_leaf: 8
> - max_depth: 4
> - Feature names: user_yelping_since_year, user_review_count, user_fans, user_votes_funny, user_votes_useful, user_votes_cool, user_average_stars, user_tip_counts
> - Class names: normal, BEST

> **_(R1-4)_** Create a decision tree for a different VIP selection criteria than the one generated in R1-3, and compare it with the decision tree created in R1-3. You can choose to add or remove additional input features other than those used in R1-3, or modify other attributes of the decision tree. The report should include the following information:
> - Input features used in the decision tree
> - Node impurity criterion
> - Attributes provided to sklearn.tree.DecisionTreeClassifier (e.g., max_depth=3)

### <ins>PART II. Association Analysis</ins>
In Part II, the goal is to perform association analysis among the attributes of website A. In R2-1 and R2-2, the objective is to execute queries using mysql-connector-python for the purpose of generating views for association analysis and recommendation systems. In R2-3, the goal is to return the results of a horizontal table from the view created in R2-2 for association analysis. In R2-4, the objective is to perform association analysis using the mlxtend library in Python and output the results.

> **_(R2-1)_** In this association analysis, focus on the top 30 categories with significant influence based on scores derived from the number of reviews, the number of businesses, and the average rating. To facilitate the analysis, create a view named "category_score" that includes the following columns, all in a single SQL statement:
> - category_id: The ID of the category
> - category_name: The name of the category
> - num_business: The number of businesses in that category
> - num_reviews: The sum of reviews for businesses in that category
> - ctg_avg_stars: The average rating of businesses in that category
> - score: The value of ctg_avg_stars multiplied by (num_reviews / num_business)
> 
> The view should be sorted in descending order based on the "score" column, and only the top 30 records based on the score should be saved in the view. Save this view as DMA_project2_team##_part2_category.csv.

> **_(R2-2)_** To facilitate association analysis and recommendation systems, the level of interest each user has for a category should be defined as a rating. The rating is defined as follows. However, for categories that only include businesses where the user has not performed any review activity, the rating should be considered as not given, rather than 0.
> 
> **Rating Equation**  rating(user, category) = 2 * (number of businesses in that category reviewed by the user with a rating of 4 or higher) + min(number of businesses reviewed by the user in that category, 5)
> 
> For example, if user "a" has a history of using 10 businesses belonging to category_b, and the user has given a rating of 4 or higher to 2 businesses in that category, the rating(user_a, category_b) would be calculated as 2 * 2 + min(10, 5), resulting in an interest level of 9 for that category.
> 
> In R2-1, create a view named "user_category_rating" for the total of 30 bundles defined, containing the rating information of users. This should be done using a single SQL statement. The view should only store categories for the 30 bundles defined in R2-1 and include rating information for user-bundle sets that have ratings.
> 
> Create a view named "user_category_rating" that includes the following columns:
> - user: user_id
> - category: Category name (one of the top 30 categories)
> - rating: Interest level defined as mentioned above
> 
> After creating the above view, create another view named "partial_user_category_rating" that only includes information for users who have 10 or more rating information. The "partial_user_category_rating" view should have the same columns as "user_category_rating". Save this "partial_user_category_rating" view as DMA_project2_team##_part2_UCR.csv.

> **_(R2-3)_** To perform association analysis, transform the partial_user_category_rating from a vertical table to a horizontal table and save the result as a pandas DataFrame. The DataFrame should have user id as the index and the bundle names as column names. If partial_user_category_rating contains rating information for a user and a category, store 1; otherwise, store 0. Each user in the stored DataFrame will play the role of a transaction in the association analysis, and each category will serve as an item.
> 
> You can use an SQL query statement or the pandas library to create the horizontal table. Save the resulting horizontal table as DMA_project2_team##_part2_horizontal.pkl.

> **_(R2-4)_** Using the DataFrame created in R2-3, generate frequent itemsets that satisfy the following conditions and perform association analysis. Additionally, provide a brief qualitative and quantitative evaluation of the results.
> - Minimum support for frequent itemsets: 0.15
> - Association analysis metric: lift (output itemsets with lift values greater than or equal to 3)
> 
> Save the results as DMA_project2_team##_part2_association.csv & .pkl.

### <ins>PART III. Recommendation System</ins>
