# Churn Dashboard Interview Questions

Here are 10 interview questions and answers based on the churn dashboard. These questions cover basic data interpretation, business insights, and critical thinking.

### 10 Interview Questions and Answers

**Q1: What are the key top-line metrics displayed on this dashboard, and what do they tell us about the overall health of the customer base?**
**Answer:** The dashboard shows 51,046 Total Customers, a Churn Rate of 28.8%, and an Average Monthly Revenue of $57.91. A churn rate of nearly 29% is relatively high for most subscription or telecom businesses, indicating a significant retention problem that needs addressing.

**Q2: Looking at the "Retention Team Impact on Churn" chart, what counterintuitive trend do you see, and what is the most likely business explanation for it?**
**Answer:** The chart shows that customers who called the retention team have a much higher churn rate (45.0%) compared to those where no call was made (28.2%). The likely explanation is "selection bias"—customers only call the retention team when they are already highly dissatisfied and intend to cancel. It doesn't necessarily mean the retention team is causing the churn, but it does highlight that the team's current save offers or strategies are not highly effective at rescuing these at-risk accounts.

**Q3: Based on the "Number of Records" by Tenure chart, during which phase of the customer lifecycle is the company managing the largest volume of customers?**
**Answer:** The company has the largest volume of customers in the "1 - Growing (12-24m)" phase, with over 22,000 total records (15,665 retained + 7,319 churned). This suggests the company is good at acquiring and keeping customers for the first year, but faces a critical drop-off or testing period between years 1 and 2. 

**Q4: How does a customer's credit rating seem to correlate with their likelihood to churn?**
**Answer:** Interestingly, the dashboard indicates that customers with better credit ratings are more likely to churn. The highest churn rates are in the "3 - Good" (31.0%) and "1 - Highest" (30.8%) categories, while the lowest churn rate is in the "5 - Low" category at 22.1%. This could imply that customers with higher credit scores have more market mobility, better alternative options, or are targeted more aggressively by competitors.

**Q5: Is there a clear, linear relationship between income group and churn rate? Please provide examples from the visual.**
**Answer:** No, the relationship is non-linear and mixed. For example, the highest churn is seen in the "0 - Lowest" income group (30.2%), but the second highest is the "7 - Very High" group (29.8%). Conversely, the lowest churn is found in the middle-tier "3 - Below Avg" group (26.3%). This suggests different income brackets churn for entirely different reasons (e.g., affordability for low-income vs. better premium offers for high-income).

**Q6: Using the top-line metrics provided, what is the approximate Monthly Recurring Revenue (MRR) lost to churned customers?**
**Answer:** First, calculate the number of churned customers: 51,046 * 28.8% ≈ 14,701 churned customers. Then multiply by the Average Monthly Revenue: 14,701 * $57.91 ≈ $851,335. The company is losing roughly $851K in revenue per month due to churn.

**Q7: Which customer tenure segment appears to be the most stable and loyal?**
**Answer:** The "3 - Loyal (48m+)" segment. Although they represent a very small absolute number of customers compared to the other groups, the proportion of churned (light blue) to retained (dark blue) customers in that specific bar is visually the smallest, indicating a very low risk of churn once a customer reaches the 4-year mark.

**Q8: If you were the Data Analyst presenting this dashboard to the VP of Marketing, what actionable recommendation would you make based on the Credit Rating and Income Group data?**
**Answer:** I would recommend shifting some acquisition marketing spend away from purely targeting "High Credit / Very High Income" individuals, as they are surprisingly flighty and have high churn rates. Instead, we should look into designing campaigns that target "Low Credit" or "Below Avg Income" groups, as they demonstrate higher loyalty (lower churn) to our service, potentially increasing our lifetime value (LTV) despite having less disposable income.

**Q9: What is a potential flaw in how the "Churn by Credit Rating" chart is formatted, and how would you fix it?**
**Answer:** The chart is currently sorted descending by Churn Rate %. Because of this, the ordinal categories of Credit Rating (Highest, High, Good, Medium, Low) are entirely out of order on the y-axis, making it difficult for the user to spot a trend quickly. I would re-sort the y-axis logically from "1 - Highest" down to "7 - Lowest" so the viewer can immediately see how churn behaves as credit rating decreases.

**Q10: Based on the "Retention Team Impact" chart, what would be your immediate next step as a manager of that team?**
**Answer:** Since a 45% churn rate on intervention calls is very high, my immediate next step would be to pull the call logs and transcripts for the "Called Retention" group. I would want to perform a text analysis or QA review to understand exactly why these customers are calling, what offers the agents are presenting, and why those offers are being rejected at such a high rate.
