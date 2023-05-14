# 2021-DMA-projects
## General Info
Projects done with my SNU colleagues in the purpose to study DMA (Data Management & Analysis). Three projects were carried out in this study, as following.

**PROJECT #1** - *Conceptual DB design & DB implementation*  
**PROJECT #2** - *DB mining & Recommendation system*  
**PROJECT #3** - *Document search engine & Classification and Clustering*  

## Project Info
### Project #1
Website A is a platform where customers write reviews about businesses. The website's customers (users) can leave reviews on various types of businesses they have used and interact with other customers by sharing reviews. This project aims to create a visualization of the ER diagram, and implementation of the database used by webite A. The project is divided into two main parts.  

**PART I** - *ER diagram Visulization*  
**PART II** - *DB Implementation & Data Entry*  

<ins>**PART I. ER Diagram Visualization**</ins>  
PART I aims to visualize the ER diagram for the database used by website A. website A's database should satisfy the following requirements:

> **_(R1-1)_** Website A stores information about registered users (customers) on the site. Users can write reviews or tips after using a business. Customer information should include a unique ID assigned by the site, the start date of site usage, the average rating of reviews they have written, the number of times other users have rated their reviews as interesting, useful, or accurate, and the number of users who have marked them as a fan.

> **_(R1-2)_** Website A stores information about registered businesses on the site. Business information should include a unique ID for the business, its current operational status, the location, the average rating received from customers, and the number of customer reviews received. Site A also has category information and feature information for each business. The category information represents the type of business, and the feature information indicates the presence or absence of specific attributes (e.g., Wi-Fi availability). Feature information may have values representing the level or extent of the attribute. Category information, feature information, and feature values should also be stored.

> **_(R1-3)_** Site A stores information about time and moments. Therefore, year information and day of the week information should be stored. Time-related information includes the opening and closing hours of businesses and the year of elite customer selection.

> **_(R1-4)_** Site A stores information about reviews left by users for businesses. Review information should include a unique identifier for the review, the business ID, the customer ID, the rating of the review, the degree to which the review is rated as interesting, useful, or accurate, and the length of the review. Site A also stores tips, which are more concise forms of reviews. Tip information should include the business ID, the customer ID, the number of likes received, and the length of the tip.

> **_(R1-5)_** Site A stores additional information about customers. Additional information includes the timing of elite customer selection and the amount of praise received by customers (such as photos) and the number of occurrences.
