# Executive Summary

**Business Type:** Likely a B2B (Business-to-Business) company selling products or services to other businesses. This is inferred from the 'customer_profiles' table which contains information about companies (company_name, industry, company_size, employee_count) and metrics like 'lifetime_value', 'engagement_score', and 'churn_risk'. The 'sales_transactions' table links customers to products, reinforcing this.

**Analysis Date:** October 19, 2025


## Key Findings

• 
FINDING: The top 10 customers by total revenue are:
- Company_112 (ID: C0112): $21,918.51
- Company_214 (ID: C0214): $20,895.10
- Company_182 (ID: C0182): $20,055.23
- Company_239 (ID: C0239): $19,298.46
- Company_62 (ID: C0062): $17,694.58
- Company_211 (ID: C0211): $17,067.55
- Company_498 (ID: C0498): $16,943.49
- Company_126 (ID: C0126): $16,829.30
- Company_464 (ID: C0464): $16,785.91
- Company_420 (ID: C0420): $16,590.82

FINDING: These top 10 customers collectively generate $184,078.96, which is 4.51% of the total revenue.

• FINDING: The average customer lifetime value is $46,986.76.

• --- Analyzing Customer Churn Factors ---
FINDING: 307 out of 500 customers (61.40%) are identified as 'High' churn risk.
FINDING: High churn risk customers have a significantly lower average engagement score (18.21) compared to Low/Medium risk customers (71.43).
FINDING: High churn risk customers have a significantly lower average Lifetime Value ($17,054.69) compared to Low/Medium risk customers ($94,598.92).

FINDING: Top industries where 'High' churn risk customers are concentrated:
- Tech: 30.62%
- Retail: 20.52%
- Healthcare: 17.92%
- Finance: 15.96%
- Manufacturing: 14.98%

FINDING: High churn risk customers show a significantly higher average 'days since last purchase' (108 days) compared to Low/Medium risk customers (102 days). This inactivity is a strong indicator of impending churn.

FINDING: For products purchased by high churn risk customers, the average product quality score was 67.41.
FINDING: The average customer satisfaction for products purchased by high churn risk customers was 67.42.
FINDING: The average return rate for products purchased by high churn risk customers was 0.0974.

• FINDING: The top 10 most profitable products are:
- Product_Premium_127 (Product ID: P128): Total Profit = $22,983.80
- Product_Premium_83 (Product ID: P084): Total Profit = $21,917.09
- Product_Premium_193 (Product ID: P194): Total Profit = $21,440.20
- Product_Premium_72 (Product ID: P073): Total Profit = $17,988.22
- Product_Premium_41 (Product ID: P042): Total Profit = $17,848.12
- Product_Premium_9 (Product ID: P010): Total Profit = $16,456.01
- Product_Premium_134 (Product ID: P135): Total Profit = $16,145.16
- Product_Premium_49 (Product ID: P050): Total Profit = $14,683.47
- Product_Premium_181 (Product ID: P182): Total Profit = $14,566.66
- Product_Premium_18 (Product ID: P019): Total Profit = $14,539.38

• Found 1 unusual values in daily_sales_metrics.marketing_spend that may indicate data quality issues or exceptional business events

## Priority Actions

• Focus retention efforts on high-value customers identified
• Focus retention efforts on high-value customers identified

## Detailed Analysis

### Business Insights

1. **Which customers generate the most revenue?**
   - Finding: 
FINDING: The top 10 customers by total revenue are:
- Company_112 (ID: C0112): $21,918.51
- Company_214 (ID: C0214): $20,895.10
- Company_182 (ID: C0182): $20,055.23
- Company_239 (ID: C0239): $19,298.46
- Company_62 (ID: C0062): $17,694.58
- Company_211 (ID: C0211): $17,067.55
- Company_498 (ID: C0498): $16,943.49
- Company_126 (ID: C0126): $16,829.30
- Company_464 (ID: C0464): $16,785.91
- Company_420 (ID: C0420): $16,590.82

FINDING: These top 10 customers collectively generate $184,078.96, which is 4.51% of the total revenue.


2. **What is the average customer lifetime value?**
   - Finding: FINDING: The average customer lifetime value is $46,986.76.


3. **Why are customers leaving?**
   - Finding: --- Analyzing Customer Churn Factors ---
FINDING: 307 out of 500 customers (61.40%) are identified as 'High' churn risk.
FINDING: High churn risk customers have a significantly lower average engagement score (18.21) compared to Low/Medium risk customers (71.43).
FINDING: High churn risk customers have a significantly lower average Lifetime Value ($17,054.69) compared to Low/Medium risk customers ($94,598.92).

FINDING: Top industries where 'High' churn risk customers are concentrated:
- Tech: 30.62%
- Retail: 20.52%
- Healthcare: 17.92%
- Finance: 15.96%
- Manufacturing: 14.98%

FINDING: High churn risk customers show a significantly higher average 'days since last purchase' (108 days) compared to Low/Medium risk customers (102 days). This inactivity is a strong indicator of impending churn.

FINDING: For products purchased by high churn risk customers, the average product quality score was 67.41.
FINDING: The average customer satisfaction for products purchased by high churn risk customers was 67.42.
FINDING: The average return rate for products purchased by high churn risk customers was 0.0974.


4. **Which products are most profitable?**
   - Finding: FINDING: The top 10 most profitable products are:
- Product_Premium_127 (Product ID: P128): Total Profit = $22,983.80
- Product_Premium_83 (Product ID: P084): Total Profit = $21,917.09
- Product_Premium_193 (Product ID: P194): Total Profit = $21,440.20
- Product_Premium_72 (Product ID: P073): Total Profit = $17,988.22
- Product_Premium_41 (Product ID: P042): Total Profit = $17,848.12
- Product_Premium_9 (Product ID: P010): Total Profit = $16,456.01
- Product_Premium_134 (Product ID: P135): Total Profit = $16,145.16
- Product_Premium_49 (Product ID: P050): Total Profit = $14,683.47
- Product_Premium_181 (Product ID: P182): Total Profit = $14,566.66
- Product_Premium_18 (Product ID: P019): Total Profit = $14,539.38


5. **Anomalies Detected in daily_sales_metrics.marketing_spend**
   - Finding: Found 1 unusual values in daily_sales_metrics.marketing_spend that may indicate data quality issues or exceptional business events

6. **Anomalies Detected in daily_sales_metrics.quantity**
   - Finding: Found 11 unusual values in daily_sales_metrics.quantity that may indicate data quality issues or exceptional business events

7. **Anomalies Detected in daily_sales_metrics.revenue**
   - Finding: Found 29 unusual values in daily_sales_metrics.revenue that may indicate data quality issues or exceptional business events

8. **Anomalies Detected in daily_sales_metrics.cost**
   - Finding: Found 9 unusual values in daily_sales_metrics.cost that may indicate data quality issues or exceptional business events

9. **Anomalies Detected in daily_sales_metrics.profit**
   - Finding: Found 17 unusual values in daily_sales_metrics.profit that may indicate data quality issues or exceptional business events

10. **Anomalies Detected in customer_profiles.employee_count**
   - Finding: Found 41 unusual values in customer_profiles.employee_count that may indicate data quality issues or exceptional business events

11. **Anomalies Detected in customer_profiles.lifetime_value**
   - Finding: Found 41 unusual values in customer_profiles.lifetime_value that may indicate data quality issues or exceptional business events

12. **Anomalies Detected in product_information.price**
   - Finding: Found 24 unusual values in product_information.price that may indicate data quality issues or exceptional business events

13. **Anomalies Detected in sales_transactions.quantity**
   - Finding: Found 18 unusual values in sales_transactions.quantity that may indicate data quality issues or exceptional business events

14. **Anomalies Detected in sales_transactions.gross_amount**
   - Finding: Found 96 unusual values in sales_transactions.gross_amount that may indicate data quality issues or exceptional business events

15. **Anomalies Detected in sales_transactions.discount_amount**
   - Finding: Found 119 unusual values in sales_transactions.discount_amount that may indicate data quality issues or exceptional business events

16. **Anomalies Detected in sales_transactions.net_amount**
   - Finding: Found 100 unusual values in sales_transactions.net_amount that may indicate data quality issues or exceptional business events

17. **Significant relationship: marketing_spend -> quantity**
   - Finding: Statistical analysis reveals a strong causal relationship between marketing_spend and quantity (p-value: 0.0000)

18. **Significant relationship: marketing_spend -> revenue**
   - Finding: Statistical analysis reveals a strong causal relationship between marketing_spend and revenue (p-value: 0.0000)

19. **Significant relationship: marketing_spend -> cost**
   - Finding: Statistical analysis reveals a strong causal relationship between marketing_spend and cost (p-value: 0.0000)

20. **Significant relationship: marketing_spend -> profit**
   - Finding: Statistical analysis reveals a moderate causal relationship between marketing_spend and profit (p-value: 0.0154)

21. **Significant relationship: customer_traffic -> quantity**
   - Finding: Statistical analysis reveals a moderate causal relationship between customer_traffic and quantity (p-value: 0.0227)

22. **Significant relationship: customer_traffic -> revenue**
   - Finding: Statistical analysis reveals a strong causal relationship between customer_traffic and revenue (p-value: 0.0000)

23. **Significant relationship: customer_traffic -> cost**
   - Finding: Statistical analysis reveals a strong causal relationship between customer_traffic and cost (p-value: 0.0061)

24. **Significant relationship: customer_traffic -> profit**
   - Finding: Statistical analysis reveals a strong causal relationship between customer_traffic and profit (p-value: 0.0000)

25. **Significant relationship: price -> quantity**
   - Finding: Statistical analysis reveals a strong causal relationship between price and quantity (p-value: 0.0000)

26. **Significant relationship: price -> month**
   - Finding: Statistical analysis reveals a moderate causal relationship between price and month (p-value: 0.0128)

27. **Significant relationship: price -> year**
   - Finding: Statistical analysis reveals a moderate causal relationship between price and year (p-value: 0.0139)

28. **Significant relationship: quantity -> revenue**
   - Finding: Statistical analysis reveals a strong causal relationship between quantity and revenue (p-value: 0.0000)

29. **Significant relationship: quantity -> cost**
   - Finding: Statistical analysis reveals a moderate causal relationship between quantity and cost (p-value: 0.0204)

30. **Significant relationship: quantity -> profit**
   - Finding: Statistical analysis reveals a strong causal relationship between quantity and profit (p-value: 0.0005)

31. **Significant relationship: revenue -> cost**
   - Finding: Statistical analysis reveals a strong causal relationship between revenue and cost (p-value: 0.0000)

32. **Significant relationship: revenue -> profit**
   - Finding: Statistical analysis reveals a strong causal relationship between revenue and profit (p-value: 0.0019)

33. **Significant relationship: revenue -> month**
   - Finding: Statistical analysis reveals a strong causal relationship between revenue and month (p-value: 0.0006)

34. **Significant relationship: revenue -> quarter**
   - Finding: Statistical analysis reveals a strong causal relationship between revenue and quarter (p-value: 0.0024)

35. **Significant relationship: revenue -> year**
   - Finding: Statistical analysis reveals a strong causal relationship between revenue and year (p-value: 0.0003)

36. **Significant relationship: cost -> profit**
   - Finding: Statistical analysis reveals a strong causal relationship between cost and profit (p-value: 0.0019)

37. **Significant relationship: profit -> month**
   - Finding: Statistical analysis reveals a strong causal relationship between profit and month (p-value: 0.0001)

38. **Significant relationship: profit -> quarter**
   - Finding: Statistical analysis reveals a strong causal relationship between profit and quarter (p-value: 0.0019)

39. **Significant relationship: profit -> year**
   - Finding: Statistical analysis reveals a strong causal relationship between profit and year (p-value: 0.0001)

40. **Significant relationship: month -> year**
   - Finding: Statistical analysis reveals a moderate causal relationship between month and year (p-value: 0.0245)

41. **Significant relationship: quality_score -> return_rate**
   - Finding: Statistical analysis reveals a moderate causal relationship between quality_score and return_rate (p-value: 0.0230)

42. **Significant relationship: quantity -> discount_percent**
   - Finding: Statistical analysis reveals a moderate causal relationship between quantity and discount_percent (p-value: 0.0296)

### Recommendations

1. **Focus retention efforts on high-value customers identified**
   - Rationale: 
FINDING: The top 10 customers by total revenue are:
- Company_112 (ID: C0112): $21,918.51
- Company_214 (ID: C0214): $20,895.10
- Company_182 (ID: C0182): $20,055.23
- Company_239 (ID: C0239): $19,298.46
- Company_62 (ID: C0062): $17,694.58
- Company_211 (ID: C0211): $17,067.55
- Company_498 (ID: C0498): $16,943.49
- Company_126 (ID: C0126): $16,829.30
- Company_464 (ID: C0464): $16,785.91
- Company_420 (ID: C0420): $16,590.82

FINDING: These top 10 customers collectively generate $184,078.96, which is 4.51% of the total revenue.

   - Impact: High

2. **Focus retention efforts on high-value customers identified**
   - Rationale: --- Analyzing Customer Churn Factors ---
FINDING: 307 out of 500 customers (61.40%) are identified as 'High' churn risk.
FINDING: High churn risk customers have a significantly lower average engagement score (18.21) compared to Low/Medium risk customers (71.43).
FINDING: High churn risk customers have a significantly lower average Lifetime Value ($17,054.69) compared to Low/Medium risk customers ($94,598.92).

FINDING: Top industries where 'High' churn risk customers are concentrated:
- Tech: 30.62%
- Retail: 20.52%
- Healthcare: 17.92%
- Finance: 15.96%
- Manufacturing: 14.98%

FINDING: High churn risk customers show a significantly higher average 'days since last purchase' (108 days) compared to Low/Medium risk customers (102 days). This inactivity is a strong indicator of impending churn.

FINDING: For products purchased by high churn risk customers, the average product quality score was 67.41.
FINDING: The average customer satisfaction for products purchased by high churn risk customers was 67.42.
FINDING: The average return rate for products purchased by high churn risk customers was 0.0974.

   - Impact: High

## Data Analyzed
- **daily_sales_metrics**: 1,095 records
- **customer_profiles**: 500 records
- **product_information**: 200 records
- **sales_transactions**: 5,000 records

---
*This report was generated using dynamic business analysis.*
