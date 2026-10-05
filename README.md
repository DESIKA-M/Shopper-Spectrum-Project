# Shopper Spectrum: Customer Segmentation & Recommendation System

## 📌 Overview

**Shopper Spectrum** is an e-commerce analytics and recommendation project that uses customer purchase data to understand customer behaviour and provide product recommendations.

The project combines **RFM (Recency, Frequency, Monetary) analysis**, **K-Means clustering**, and **item-based collaborative filtering** to segment customers and recommend relevant products.

An interactive **Streamlit application** was developed to make the analysis and recommendations accessible through a web interface.

---

## 🎯 Objectives

- Analyze customer purchasing behaviour.
- Segment customers based on their purchasing patterns.
- Identify meaningful customer groups using machine learning.
- Build a product recommendation system based on purchasing behaviour.
- Provide an interactive interface for viewing customer segments and recommendations.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib**

### Machine Learning Techniques

- RFM Analysis
- K-Means Clustering
- Elbow Method
- Silhouette Score
- Cosine Similarity
- Item-Based Collaborative Filtering

---

## 📊 Dataset

The project uses an **Online Retail transaction dataset** containing customer purchase information such as:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

The original dataset contains **541,909 transaction records**.

After data cleaning, **397,884 records** were retained for further analysis.

---

## 🔄 Project Workflow

```text
Raw Online Retail Dataset
          ↓
     Data Cleaning
          ↓
   Feature Engineering
          ↓
      RFM Analysis
          ↓
    Data Standardization
          ↓
      K-Means Clustering
          ↓
 Customer Segmentation
          ↓
 Product Recommendation
          ↓
    Streamlit Application
```

---

## 🧹 Data Preprocessing

The raw transaction data was cleaned before performing the analysis.

The preprocessing included:

- Handling missing values
- Removing invalid transactions
- Removing duplicate records
- Handling returned/cancelled transactions
- Creating useful features for analysis
- Calculating total transaction value

A **TotalPrice** feature was calculated as:

```text
TotalPrice = Quantity × UnitPrice
```

After preprocessing, the cleaned dataset was used for customer-level analysis.

---

# 👥 Customer Segmentation

## RFM Analysis

RFM analysis was used to understand customer purchasing behaviour.

### Recency

Measures **how recently a customer made a purchase**.

```text
Recency = Reference Date - Last Purchase Date
```

A lower recency value generally indicates a more recent customer purchase.

### Frequency

Measures **how frequently a customer made purchases**.

### Monetary

Measures **how much money a customer spent**.

The three values were combined to create an RFM profile for each customer.

```text
Customer
   ↓
Recency
Frequency
Monetary
   ↓
RFM Profile
```

---

## 🤖 K-Means Clustering

K-Means clustering was used to group customers with similar purchasing behaviour.

Before applying K-Means, the RFM features were standardized using **StandardScaler**.

The **Elbow Method** was used to determine an appropriate number of clusters.

The project selected:

```text
K = 4
```

The resulting customer groups were interpreted as:

- **High-Value Customers**
- **Regular Customers**
- **Occasional Customers**
- **At-Risk Customers**

The clustering performance was also evaluated using the **Silhouette Score**, which was approximately:

```text
0.616
```

---

# 🛍️ Product Recommendation System

The project also includes a product recommendation system based on **item-based collaborative filtering**.

A customer-product purchase matrix was created using:

```text
CustomerID × StockCode
```

The matrix was then transposed so that products could be compared with each other.

### Cosine Similarity

Cosine similarity was used to measure the similarity between products based on their purchasing patterns.

For example:

```text
Product A
     ↓
Compare with other products
     ↓
Product B → 0.91
Product C → 0.72
Product D → 0.15
```

Products with higher similarity scores are considered more similar and can be recommended to customers.

The system generates **top product recommendations** based on these similarity scores.

---

# 🌐 Streamlit Application

A Streamlit application was developed to provide an interactive interface for the project.

The application acts as the **UI/application layer** for the Python-based analytics and recommendation logic.

Users can interact with the application to:

- Select or enter customer information
- View the customer's predicted segment
- Generate product recommendations
- Explore the results without directly running the Python code

The application can be launched using:

```bash
streamlit run app.py
```

---

## 📈 Results

The project successfully:

- Cleaned and prepared the retail transaction data.
- Created customer-level RFM features.
- Segmented customers into **4 behavioural groups** using K-Means.
- Achieved a silhouette score of approximately **0.616**.
- Built an item-based recommendation system using cosine similarity.
- Developed an interactive Streamlit application for accessing the results.

---

## 💡 Business Applications

The system can help an e-commerce business:

### Customer Segmentation

Identify different types of customers and understand their purchasing behaviour.

### Customer Retention

Identify **At-Risk customers** and target them with appropriate campaigns.

### Customer Loyalty

Identify **High-Value customers** and provide personalized offers or loyalty benefits.

### Product Recommendations

Recommend relevant products to customers based on purchasing patterns.

---

## 🚀 Future Enhancements

- Integrate real-time e-commerce transaction data.
- Improve recommendations using hybrid recommendation techniques.
- Add more customer behavioural features.
- Deploy the Streamlit application online.
- Add real-time dashboards and business KPIs.
- Incorporate more advanced recommendation models.

---

## 👩‍💻 Author

**DESIKA M**

B.Tech – Electronics and Communication Engineering  
SRM Institute of Science and Technology
