WITH last_order AS (
    SELECT
        c.customer_unique_id,
        MAX(o.order_purchase_timestamp) AS last_purchase_date
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_unique_id
),
customer_metrics AS (
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(oi.price + oi.freight_value) AS total_spend,
        AVG(oi.price + oi.freight_value) AS avg_spend_per_order,
        AVG(r.review_score) AS avg_review_score,
        COUNT(DISTINCT p.payment_type) AS num_payment_methods,
        -- Replace JULIANDAY with DATEDIFF for customer tenure
        DATEDIFF(MAX(o.order_purchase_timestamp), MIN(o.order_purchase_timestamp)) AS customer_tenure_days
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN order_reviews r ON o.order_id = r.order_id
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)

SELECT
    cm.*,
    lo.last_purchase_date,
    CASE
        WHEN DATEDIFF(
            (SELECT MAX(order_purchase_timestamp) FROM orders),
            lo.last_purchase_date
        ) > 180 THEN 1
        ELSE 0
    END AS churned
FROM customer_metrics cm
JOIN last_order lo ON cm.customer_unique_id = lo.customer_unique_id;
