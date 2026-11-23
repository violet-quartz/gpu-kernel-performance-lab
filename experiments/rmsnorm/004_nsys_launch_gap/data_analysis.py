import sqlite3
import statistics

def stat_indicator(rows, name):
    print(f"==========={name}==========")
    data = [
        row[name] for row in rows if row[name] is not None
    ]
    print("Count:", len(data))
    print("Mean:", statistics.mean(data))
    print("Median:", statistics.median(data))
    print("Min:", min(data))
    print("Max:", max(data))
    print("P90:", sorted(data)[int(0.9 * (len(data) - 1))])

def stat_normal_kernel_duration_and_gap(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    sql = """
    WITH normal_range AS (
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = 'normal_launch'
    ),
    normal_kernels AS (
        SELECT k.*
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN normal_range n
        ON k.start >= n.start AND k.end <= n.end
    ),
    ordered_kernels AS (
        SELECT
            start,
            end,
            streamId,
            (end - start) / 1000.0 AS duration_us,
            (
                LEAD(start) OVER (
                    PARTITION BY deviceId, contextId, streamId
                    ORDER BY start
                ) - end
            ) / 1000.0 AS gap_us
        FROM normal_kernels
    )
    SELECT *
    FROM ordered_kernels
    ORDER BY start;
    """

    rows = conn.execute(sql).fetchall()
    for i in range(5):
        print(dict(rows[i]))

    stat_indicator(rows, "gap_us")
    stat_indicator(rows, "duration_us")

    conn.close()

def stat_launch_api(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    sql = """
WITH normal_range AS (
    SELECT start, end
    FROM NVTX_EVENTS
    WHERE text = 'normal_launch'
),
launch_api AS (
    SELECT
        r.start,
        r.end,
        r.correlationId,
        s.value AS api_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s
      ON r.nameId = s.id
    JOIN normal_range n
      ON r.start >= n.start
     AND r.end <= n.end
    WHERE s.value = 'cuLaunchKernelEx'
)
SELECT
    start,
    end,
    correlationId,
    api_name,
    (end - start) / 1000.0 AS api_duration_us,
    (
        LEAD(start) OVER (ORDER BY start) - start
    ) / 1000.0 AS launch_interval_us,
    (
        LEAD(start) OVER (ORDER BY start) - end
    ) / 1000.0 AS between_api_us
FROM launch_api
ORDER BY start;
"""
    rows = conn.execute(sql).fetchall()
    for i in range(5):
        print(dict(rows[i]))

    stat_indicator(rows, "api_duration_us")
    stat_indicator(rows, "launch_interval_us")
    stat_indicator(rows, "between_api_us")

    conn.close()

def stat_graph_kernel_duration_and_gap(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    sql = """
WITH graph_kernels AS (
    SELECT
        start,
        end,
        deviceId,
        contextId,
        streamId,
        graphNodeId,
        graphId
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE graphNodeId IS NOT NULL
),
ordered_kernels AS (
    SELECT
        start,
        end,
        graphNodeId,
        graphId,
        streamId,
        (end - start) / 1000.0 AS duration_us,
        (
            LEAD(start) OVER (
                PARTITION BY deviceId, contextId, streamId, graphId
                ORDER BY start
            ) - end
        ) / 1000.0 AS gap_us
    FROM graph_kernels
)
SELECT *
FROM ordered_kernels
ORDER BY start;
"""
    rows = conn.execute(sql).fetchall()
    for i in range(5):
        print(dict(rows[i]))

    stat_indicator(rows, "gap_us")
    stat_indicator(rows, "duration_us")
    
    conn.close()
    

def main():
    db = "/Users/flora/Downloads/rmsnorm_launch_gap_2.sqlite"
    print("*********Normal Launch**************")
    stat_normal_kernel_duration_and_gap(db)
    print("*********Launch API*****************")
    stat_launch_api(db)
    print("*********CUDA Graph*****************")
    stat_graph_kernel_duration_and_gap(db)

if __name__ == "__main__":
    main()