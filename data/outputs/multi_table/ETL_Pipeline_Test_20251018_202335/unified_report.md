# Multi-Table Discovery Report

**Analysis:** ETL Pipeline Test
**Date:** 2025-10-18 20:23
**Datasets:** 3

## Datasets Analyzed
- **sales_transactions** (Sales domain)
  - 6 rows × 7 columns
  - This dataset represents sales transactions, including customer, product, date, revenue, cost, and quantity.
- **customer_lifetime_values** (Sales domain)
  - 5 rows × 7 columns
  - This dataset contains information about customer lifetime values, including customer demographics, acquisition details, and their calculated lifetime value.
- **products** (Product domain)
  - 3 rows × 6 columns
  - This dataset contains information about products, including their names, categories, prices, and launch dates. It can be used to analyze product performance and trends.

## Single-Table Insights

### sales_transactions
- **Insight 1: Product P001 is the Top Performer in Revenue and Quantity, but P002 has the Highest Profit Margin**
  - ### Insight 1: Product P001 is the Top Performer in Revenue and Quantity, but P002 has the Highest Profit Margin
Product P001 leads in total revenue with $3300 and total quantity sold with 33 units. However, Product P002, despite having the lowest total revenue ($1500) and quantity (15 units), boasts the highest average profit margin at 40.00%. Product P003 is a close second in revenue ($2900) and quantity (29 units), with a profit margin of 32.76%. P001 has the lowest average profit margin at 31.71%. This highlights a crucial trade-off between sales volume and profitability. While P001 is a strong revenue driver, P002 is significantly more profitable per unit. The business should consider strategies to increase sales of P002, perhaps through targeted marketing or bundling, to capitalize on its high margin. For P001, investigating ways to reduce costs or increase its selling price could improve its profitability without sacrificing its strong sales performance.
(Visualizations: `insight_1_product_revenue_bar.png`, `insight_1_product_profit_margin_bar.png`)
- **Insight 2: Customer C001 is the Highest Revenue Generator and Most Frequent Purchaser**
  - ### Insight 2: Customer C001 is the Highest Revenue Generator and Most Frequent Purchaser
Customer C001 is the top revenue generator, contributing $2800, and also has the highest transaction count with 2 purchases. Other customers (C002, C003, C004, C005) have only made one transaction each. Customer C005, despite only one transaction, generated the second-highest revenue ($1800) and the highest profit ($1200). Customer C001 shows repeat purchase behavior, indicating loyalty or satisfaction. This customer should be a focus for retention strategies and potential upselling/cross-selling. Customer C005, while a one-time purchaser in this dataset, represents a high-value transaction. Understanding what drove C005's large purchase could inform strategies to attract similar high-value customers or encourage C005 to make repeat purchases.
(Visualizations: `insight_2_customer_revenue_bar.png`, `insight_2_customer_transactions_bar.png`)
- **Insight 3: Sales are Concentrated on Sundays, with January being the Dominant Month**
  - ### Insight 3: Sales are Concentrated on Sundays, with January being the Dominant Month
All transactions in the dataset occurred on a Sunday, generating $9800 in revenue. This indicates a strong weekly sales pattern. Monthly analysis shows that January accounted for $6000 in revenue, significantly more than February's $3800, suggesting a potential decline or seasonality after the initial month. The business has a clear weekly sales peak on Sundays. This information is critical for staffing, inventory management, and marketing efforts. Promotions or special offers could be concentrated on Sundays to maximize sales, or efforts could be made to spread sales throughout the week. The drop in revenue from January to February suggests potential seasonality or a one-time surge in January. Further investigation into the cause of this monthly variation is needed to understand if it's a trend or an anomaly, and to plan for future sales cycles.
(Visualizations: `insight_3_revenue_by_day_of_week.png`, `insight_3_revenue_by_month.png`)
- **Insight 4: Strong Positive Correlation Between Revenue, Cost, Profit, and Quantity**
  - ### Insight 4: Strong Positive Correlation Between Revenue, Cost, Profit, and Quantity
There is an extremely strong positive correlation (close to 1.0) between `revenue`, `cost`, `quantity`, and `profit`. This indicates that as the quantity sold increases, revenue, cost, and profit all increase proportionally. Interestingly, `profit_margin` shows very low correlation with these variables, suggesting that the percentage profitability is not directly tied to the volume of sales or absolute revenue/cost/profit in this dataset. This strong correlation confirms that the business model is volume-driven. To increase total revenue and profit, the primary focus should be on increasing the quantity of products sold. The low correlation with profit margin implies that while sales volume drives absolute profit, it doesn't necessarily improve the *efficiency* of profit generation (i.e., the percentage of revenue kept as profit). This reinforces the importance of managing product-specific profit margins (as seen in Insight 1) independently of overall sales volume.
(Visualizations: `insight_4_correlation_heatmap.png`, `insight_4_revenue_quantity_scatter.png`)
- **Insight 5: Average Profit Margin Varies Significantly Across Products, Indicating Different Profitability Profiles**
  - ### Insight 5: Average Profit Margin Varies Significantly Across Products, Indicating Different Profitability Profiles
Product P002 stands out with a significantly higher average profit margin of 40.00%, compared to P003 (32.76%) and P001 (31.71%). The overall average profit margin across all products is 33.55%. This indicates that P002 is considerably more efficient at converting revenue into profit than the other products. Understanding the varying profit margins per product is crucial for strategic decision-making. The business should prioritize efforts to promote and sell P002, as each sale contributes more to the bottom line. For products P001 and P003, management should investigate cost reduction opportunities or consider price adjustments to improve their profitability, especially for P001 which has the lowest margin. This segmentation allows for more targeted and effective product management strategies.
(Visualizations: `insight_5_avg_profit_margin_product_bar.png`)

### customer_lifetime_values
- **Insight 1: Tech Industry Drives Highest Total CLV, While Finance Leads in Average CLV**
  - ### Insight 1: Tech Industry Drives Highest Total CLV, While Finance Leads in Average CLV

The 'Tech' industry contributes the most to the total customer lifetime value, with a combined sum of **$110,000** from two customers. This indicates a strong overall contribution from this sector. However, when looking at the average lifetime value per customer, the 'Finance' industry stands out with the highest average of **$75,000**, albeit from a single customer in this dataset. The 'Healthcare' industry shows the lowest total and average CLV at **$35,000**.

This insight suggests that while the Tech industry has a larger customer base (2 customers vs 1 for Finance, Retail, Healthcare) and thus a higher aggregate CLV, individual customers in the Finance sector might be more valuable on average. Businesses might consider focusing on acquiring more high-value customers in the Finance sector or increasing the average value of customers in the Tech sector.

**Supporting Visualizations:**
- `insight_1_total_clv_by_industry.png`: Bar chart showing total lifetime value by industry.
- `insight_1_average_clv_by_industry.png`: Bar chart showing average lifetime value by industry.
- **Insight 2: North Region Leads in Total CLV, While South Region Has Highest Average CLV**
  - ### Insight 2: North Region Leads in Total CLV, While South Region Has Highest Average CLV

The 'North' region contributes the most to the total customer lifetime value, with a combined sum of **$85,000** from two customers. This indicates a significant overall contribution from this geographical area. However, when examining the average lifetime value per customer, the 'South' region shows the highest average of **$75,000**, though this is based on a single customer in the dataset. The 'East' region has the lowest total and average CLV at **$45,000**.

This insight suggests that while the North region has a larger customer presence and higher aggregate CLV, individual customers in the South region might be more valuable on average. Businesses could consider strategies to replicate the success seen in the South region's customer acquisition or retention, or to increase the average value of customers in the North region.

**Supporting Visualizations:**
- `insight_2_total_clv_by_region.png`: Bar chart showing total lifetime value by region.
- `insight_2_average_clv_by_region.png`: Bar chart showing average lifetime value by region.
- **Insight 3: Q1 Acquisitions Drive Highest Total and Average CLV**
  - ### Insight 3: Q1 Acquisitions Drive Highest Total and Average CLV

Customers acquired in **Quarter 1** (January-March) collectively contribute the most to total customer lifetime value, with a sum of **$125,000** from two customers. This quarter also boasts the highest average CLV per customer at **$62,500**. Specifically, March (Month 3) stands out with the highest individual customer CLV of **$75,000**. In contrast, customers acquired in **Quarter 4** (October-December), particularly in November (Month 11), show the lowest total and average CLV at **$35,000**.

This insight suggests that the first quarter of the year is a particularly fruitful period for acquiring high-value customers. Businesses could consider allocating more marketing and sales resources to Q1 to capitalize on this trend, and investigate why Q4 acquisitions tend to yield lower CLV to potentially improve strategies during that period.

**Supporting Visualizations:**
- `insight_3_total_clv_by_month.png`: Bar chart showing total lifetime value by acquisition month.
- `insight_3_total_clv_by_quarter.png`: Bar chart showing total lifetime value by acquisition quarter.
- **Insight 4: Finance in South Region and Tech in West Region are Top Performing Segments by CLV**
  - ### Insight 4: Finance in South Region and Tech in West Region are Top Performing Segments by CLV

The most valuable customer segment, when considering both industry and region, is **Finance customers in the South region**, contributing **$75,000** in lifetime value from a single customer. Following closely is the **Tech industry in the West region**, with a CLV of **$60,000** from one customer. Conversely, the **Healthcare industry in the North region** represents the lowest performing segment in this dataset, with a CLV of **$35,000**.

This insight highlights specific high-value niches that the business could target for further growth or analyze to understand the success factors. It also points to areas where performance is lower, prompting investigation into potential improvements or different strategies.

**Supporting Visualizations:**
- `insight_4_total_clv_by_industry_region.png`: Bar chart showing total lifetime value by industry and region segment.

### products