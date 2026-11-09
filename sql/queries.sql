
-- Q1: Customers with qualifying completed-order spend above ₹8,000
SELECT
    customer_id,
    SUM(order_amount) AS total_spend,
    COUNT(*) AS order_count
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Revenue by city and product category
SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS revenue,
    COUNT(*) AS completed_orders
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Products priced above the average catalogue price
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: Customers who have never placed a completed order
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Customers whose completed-order spend is above average
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    cs.total_spend,
    av.avg_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6: Customer spending ranks within each city
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
),
ranked_customers AS (
    SELECT
        customer_id,
        city,
        total_spend,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS spend_rank,
        DENSE_RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS dense_spend_rank,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC, customer_id
        ) AS row_num
    FROM customer_spend
)
SELECT *
FROM ranked_customers
WHERE spend_rank = 1
ORDER BY city;


-- Q7: Daily completed revenue and cumulative revenue
WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS order_day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    order_day,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_day;
