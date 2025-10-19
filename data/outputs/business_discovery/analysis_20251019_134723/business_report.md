# Executive Summary

**Business Type:** The business appears to be a B2B (Business-to-Business) company selling a range of products to other companies. The `customer_profiles` table suggests they acquire customers and track their lifetime value and churn risk. The `product_information` table indicates different product categories with varying price points, quality scores, and profit margins. The daily sales metrics provide an overview of operational performance.

**Analysis Date:** October 19, 2025


## Key Findings

• Using detailed sales transactions and product information to determine most profitable products.

FINDING: Top 5 Most Profitable Products (calculated from actual sales transactions and product margins):
- Product_Premium_127 (ID: P128): Total Profit = $22,983.80
- Product_Premium_83 (ID: P084): Total Profit = $21,917.09
- Product_Premium_193 (ID: P194): Total Profit = $21,440.20
- Product_Premium_72 (ID: P073): Total Profit = $17,988.22
- Product_Premium_41 (ID: P042): Total Profit = $17,848.12

• --- Analyzing Customer Churn Risk by Segments ---

Analyzing churn risk by 'industry':
FINDING: Top 3 segments within 'industry' with highest churn risk:
     industry  total_customers  high_churn_risk_customers  high_churn_risk_percentage
         Tech              133                         94                   70.676692
       Retail               94                         63                   67.021277
Manufacturing               77                         46                   59.740260
FINDING: The 'industry' segment 'Tech' has the highest churn risk at 70.68% (affecting 94 out of 133 customers).

Analyzing churn risk by 'company_size':
FINDING: Top 3 segments within 'company_size' with highest churn risk:
company_size  total_customers  high_churn_risk_customers  high_churn_risk_percentage
       Small              191                        191                   100.00000
      Medium              165                        116                    70.30303
FINDING: The 'company_size' segment 'Small' has the highest churn risk at 100.00% (affecting 191 out of 191 customers).

Analyzing churn risk by 'region':
FINDING: Top 3 segments within 'region' with highest churn risk:
region  total_customers  high_churn_risk_customers  high_churn_risk_percentage
 South              116                         76                   65.517241
  East              120                         74                   61.666667
 North              135                         83                   61.481481
FINDING: The 'region' segment 'South' has the highest churn risk at 65.52% (affecting 76 out of 116 customers).

--- Overall Summary of Highest Churn Risk Segments Across All Types ---
FINDING: The overall highest churn risk segment is 'Small' (from 'company_size' category) with 100.00% of its customers being 'High' churn risk.

• Customer Lifetime Value by Industry:
FINDING: Industry 'Finance' has an average CLTV of $56,459.95
FINDING: Industry 'Healthcare' has an average CLTV of $54,219.04
FINDING: Industry 'Manufacturing' has an average CLTV of $45,850.26
FINDING: Industry 'Retail' has an average CLTV of $43,471.92
FINDING: Industry 'Tech' has an average CLTV of $37,870.16

Top 5 Industries by Average Customer Lifetime Value:
- Finance: $56,459.95
- Healthcare: $54,219.04
- Manufacturing: $45,850.26
- Retail: $43,471.92
- Tech: $37,870.16

FINDING: The industry with the highest average CLTV is 'Finance' with $56,459.95.

• Analyzing marketing spend and customer acquisition using explicit customer acquisition dates and daily sales metrics...

FINDING: Monthly Marketing Spend vs. New Customers Acquired (based on explicit acquisition dates)
Correlation Coefficient: -0.1687
FINDING: There is a weak or no significant linear correlation between marketing spend and customer acquisition.

FINDING: Average Cost Per Acquisition (CPA) across available months: $24,693.00
FINDING: The most efficient month for customer acquisition was 2021-10 with a CPA of $12360.63 (Spend: $135,966.88, Acquired: 11 customers).
FINDING: The least efficient month for customer acquisition was 2023-12 with a CPA of $236201.85 (Spend: $236,201.85, Acquired: 1 customers).


## Priority Actions

• Optimize product mix to focus on high-margin items
• Focus retention efforts on high-value customers identified
• Focus retention efforts on high-value customers identified

## Detailed Analysis

### Business Insights

1. **Which products are most profitable?**
   - Finding: Using detailed sales transactions and product information to determine most profitable products.

FINDING: Top 5 Most Profitable Products (calculated from actual sales transactions and product margins):
- Product_Premium_127 (ID: P128): Total Profit = $22,983.80
- Product_Premium_83 (ID: P084): Total Profit = $21,917.09
- Product_Premium_193 (ID: P194): Total Profit = $21,440.20
- Product_Premium_72 (ID: P073): Total Profit = $17,988.22
- Product_Premium_41 (ID: P042): Total Profit = $17,848.12


2. **Which customer segments have the highest churn risk?**
   - Finding: --- Analyzing Customer Churn Risk by Segments ---

Analyzing churn risk by 'industry':
FINDING: Top 3 segments within 'industry' with highest churn risk:
     industry  total_customers  high_churn_risk_customers  high_churn_risk_percentage
         Tech              133                         94                   70.676692
       Retail               94                         63                   67.021277
Manufacturing               77                         46                   59.740260
FINDING: The 'industry' segment 'Tech' has the highest churn risk at 70.68% (affecting 94 out of 133 customers).

Analyzing churn risk by 'company_size':
FINDING: Top 3 segments within 'company_size' with highest churn risk:
company_size  total_customers  high_churn_risk_customers  high_churn_risk_percentage
       Small              191                        191                   100.00000
      Medium              165                        116                    70.30303
FINDING: The 'company_size' segment 'Small' has the highest churn risk at 100.00% (affecting 191 out of 191 customers).

Analyzing churn risk by 'region':
FINDING: Top 3 segments within 'region' with highest churn risk:
region  total_customers  high_churn_risk_customers  high_churn_risk_percentage
 South              116                         76                   65.517241
  East              120                         74                   61.666667
 North              135                         83                   61.481481
FINDING: The 'region' segment 'South' has the highest churn risk at 65.52% (affecting 76 out of 116 customers).

--- Overall Summary of Highest Churn Risk Segments Across All Types ---
FINDING: The overall highest churn risk segment is 'Small' (from 'company_size' category) with 100.00% of its customers being 'High' churn risk.


3. **What is the customer lifetime value by industry?**
   - Finding: Customer Lifetime Value by Industry:
FINDING: Industry 'Finance' has an average CLTV of $56,459.95
FINDING: Industry 'Healthcare' has an average CLTV of $54,219.04
FINDING: Industry 'Manufacturing' has an average CLTV of $45,850.26
FINDING: Industry 'Retail' has an average CLTV of $43,471.92
FINDING: Industry 'Tech' has an average CLTV of $37,870.16

Top 5 Industries by Average Customer Lifetime Value:
- Finance: $56,459.95
- Healthcare: $54,219.04
- Manufacturing: $45,850.26
- Retail: $43,471.92
- Tech: $37,870.16

FINDING: The industry with the highest average CLTV is 'Finance' with $56,459.95.


4. **How does marketing spend affect customer acquisition?**
   - Finding: Analyzing marketing spend and customer acquisition using explicit customer acquisition dates and daily sales metrics...

FINDING: Monthly Marketing Spend vs. New Customers Acquired (based on explicit acquisition dates)
Correlation Coefficient: -0.1687
FINDING: There is a weak or no significant linear correlation between marketing spend and customer acquisition.

FINDING: Average Cost Per Acquisition (CPA) across available months: $24,693.00
FINDING: The most efficient month for customer acquisition was 2021-10 with a CPA of $12360.63 (Spend: $135,966.88, Acquired: 11 customers).
FINDING: The least efficient month for customer acquisition was 2023-12 with a CPA of $236201.85 (Spend: $236,201.85, Acquired: 1 customers).


### Recommendations

1. **Optimize product mix to focus on high-margin items**
   - Rationale: Using detailed sales transactions and product information to determine most profitable products.

FINDING: Top 5 Most Profitable Products (calculated from actual sales transactions and product margins):
- Product_Premium_127 (ID: P128): Total Profit = $22,983.80
- Product_Premium_83 (ID: P084): Total Profit = $21,917.09
- Product_Premium_193 (ID: P194): Total Profit = $21,440.20
- Product_Premium_72 (ID: P073): Total Profit = $17,988.22
- Product_Premium_41 (ID: P042): Total Profit = $17,848.12

   - Impact: Medium

2. **Focus retention efforts on high-value customers identified**
   - Rationale: --- Analyzing Customer Churn Risk by Segments ---

Analyzing churn risk by 'industry':
FINDING: Top 3 segments within 'industry' with highest churn risk:
     industry  total_customers  high_churn_risk_customers  high_churn_risk_percentage
         Tech              133                         94                   70.676692
       Retail               94                         63                   67.021277
Manufacturing               77                         46                   59.740260
FINDING: The 'industry' segment 'Tech' has the highest churn risk at 70.68% (affecting 94 out of 133 customers).

Analyzing churn risk by 'company_size':
FINDING: Top 3 segments within 'company_size' with highest churn risk:
company_size  total_customers  high_churn_risk_customers  high_churn_risk_percentage
       Small              191                        191                   100.00000
      Medium              165                        116                    70.30303
FINDING: The 'company_size' segment 'Small' has the highest churn risk at 100.00% (affecting 191 out of 191 customers).

Analyzing churn risk by 'region':
FINDING: Top 3 segments within 'region' with highest churn risk:
region  total_customers  high_churn_risk_customers  high_churn_risk_percentage
 South              116                         76                   65.517241
  East              120                         74                   61.666667
 North              135                         83                   61.481481
FINDING: The 'region' segment 'South' has the highest churn risk at 65.52% (affecting 76 out of 116 customers).

--- Overall Summary of Highest Churn Risk Segments Across All Types ---
FINDING: The overall highest churn risk segment is 'Small' (from 'company_size' category) with 100.00% of its customers being 'High' churn risk.

   - Impact: High

3. **Focus retention efforts on high-value customers identified**
   - Rationale: Customer Lifetime Value by Industry:
FINDING: Industry 'Finance' has an average CLTV of $56,459.95
FINDING: Industry 'Healthcare' has an average CLTV of $54,219.04
FINDING: Industry 'Manufacturing' has an average CLTV of $45,850.26
FINDING: Industry 'Retail' has an average CLTV of $43,471.92
FINDING: Industry 'Tech' has an average CLTV of $37,870.16

Top 5 Industries by Average Customer Lifetime Value:
- Finance: $56,459.95
- Healthcare: $54,219.04
- Manufacturing: $45,850.26
- Retail: $43,471.92
- Tech: $37,870.16

FINDING: The industry with the highest average CLTV is 'Finance' with $56,459.95.

   - Impact: High

## Data Analyzed
- **customer_profiles**: 500 records
- **product_information**: 200 records
- **sales_daily_metrics**: 1,095 records
- **sales_transactions**: 5,000 records

---
*This report was generated using dynamic business analysis.*
