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

> **_(R2-1)_**
