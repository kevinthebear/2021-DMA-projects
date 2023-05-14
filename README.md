# 2021-DMA-projects
## General Info
Projects done with my SNU colleagues in the purpose to study DMA (Data Management & Analysis). 5 groups participated in the study, each group working seperately as individual teams to carry out the project requirements. (Our group was Team 5!) Three projects were carried out in this study, as following.

**PROJECT #1** - *Conceptual DB design & DB implementation*  
**PROJECT #2** - *DB mining & Recommendation system*  
**PROJECT #3** - *Document search engine & Classification and Clustering*  

## Project Info
### Project #1 - Conceptual DB design & DB implementation
Website A is a platform where customers write reviews about businesses. The website's customers (users) can leave reviews on various types of businesses they have used and interact with other customers by sharing reviews. This project aims to create a visualization of the ER diagram, and implementation of the database used by webite A. The project is divided into two main parts.  

**PART I** - *ER diagram Visulization*  
**PART II** - *DB Implementation & Data Entry*  

<ins>**PART I. ER Diagram Visualization**</ins>  
PART I aims to visualize the ER diagram for the database used by website A. website A's database should satisfy the following requirements:

> **_(R1-1)_** Website A stores information about registered users (customers) on the site. Users can write reviews or tips after using a business. Customer information should include a unique ID assigned by the site, the start date of site usage, the average rating of reviews they have written, the number of times other users have rated their reviews as interesting, useful, or accurate, and the number of users who have marked them as a fan.

> **_(R1-2)_** Website A stores information about registered businesses on the site. Business information should include a unique ID for the business, its current operational status, the location, the average rating received from customers, and the number of customer reviews received. Site A also has category information and feature information for each business. The category information represents the type of business, and the feature information indicates the presence or absence of specific attributes (e.g., Wi-Fi availability). Feature information may have values representing the level or extent of the attribute. Category information, feature information, and feature values should also be stored.

> **_(R1-3)_** Webite A stores information about time and moments. Therefore, year information and day of the week information should be stored. Time-related information includes the opening and closing hours of businesses and the year of elite customer selection.

> **_(R1-4)_** Webite A stores information about reviews left by users for businesses. Review information should include a unique identifier for the review, the business ID, the customer ID, the rating of the review, the degree to which the review is rated as interesting, useful, or accurate, and the length of the review. Site A also stores tips, which are more concise forms of reviews. Tip information should include the business ID, the customer ID, the number of likes received, and the length of the tip.

> **_(R1-5)_** Webite A stores additional information about customers. Additional information includes the timing of elite customer selection and the amount of praise received by customers (such as photos) and the number of occurrences.

<ins>**PART II. DB Implementation & Data Entry**</ins>  
PART II aims to design a suitable database schema for the data of website A and create the corresponding database tables. The goal is to proceed with the actual data entry.

> **_(R2-1)_** Before utilizing the data from Website A, it needs to be stored in MySQL. To do this, a schema with the name DMA_team## needs to be created, where ## represents the team number. For example, the schema name for Team 1 would be DMA_team01. If the schema already exists, the creation process should not be repeated.

> **_(R2-2)_** After designing the schema, tables need to be created to store the data. The table and column names and their order should match the tables and columns in the given dataset. Columns with values of 0 or 1 should be created as TINYINT(1), INTEGER types should be created as 'INT(11)', and STRING types should be created as 'VARCHAR(255)'. If the length exceeds 255 characters, 'LONGTEXT' should be used. Dates and times should be created using 'DATETIME'. If the table already exists, the creation process should not be repeated. Foreign key constraints will not be specified in (R2-2) but will be added after data insertion in (R2-3).

> **_(R2-3)_** Data needs to be stored in the created tables. The data is provided in CSV format and should not be directly modified.

> **_(R2-4)_** Foreign key constraints should be applied to the database schema to reflect the relationships between tables.

### Project #2 - DB Mining & Recommendation System
This project aims to build a database using the provided additional data, following the approach of Project #1. It further aims to implement DB mining and an Automated Recommendation System using the constructed database. The project is divided into three main parts and requires implementation using Python and MySQL.  

**PART I** - *Decision Tree*  
**PART II** - *Association Analysis*  
**PART III** - *Recommendation System*  
