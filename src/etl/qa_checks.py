from sqlalchemy import text
from src.utils.db import get_engine

def scalar(conn, sql: str):
    return conn.execute(text(sql)).scalar()

def main():
    engine = get_engine()
    with engine.begin() as conn:
        fact_cnt = scalar(conn, "SELECT COUNT(*) FROM dwh.fact_flow;")
        dest_cnt = scalar(conn, "SELECT COUNT(*) FROM dwh.dim_destination;")
        years_cnt = scalar(conn, """
            SELECT COUNT(DISTINCT dd.year)
            FROM dwh.fact_flow ff
            JOIN dwh.dim_date dd ON dd.date_id=ff.date_id;
        """)
        null_dest = scalar(conn, "SELECT COUNT(*) FROM dwh.fact_flow WHERE destination_id IS NULL;")
        null_date = scalar(conn, "SELECT COUNT(*) FROM dwh.fact_flow WHERE date_id IS NULL;")
        min_year = scalar(conn, """
            SELECT MIN(dd.year)
            FROM dwh.fact_flow ff
            JOIN dwh.dim_date dd ON dd.date_id=ff.date_id;
        """)
        max_year = scalar(conn, """
            SELECT MAX(dd.year)
            FROM dwh.fact_flow ff
            JOIN dwh.dim_date dd ON dd.date_id=ff.date_id;
        """)

    print("QA CHECKS")
    print("fact_flow rows:", fact_cnt)
    print("dim_destination rows:", dest_cnt)
    print("years in fact:", years_cnt, f"({min_year}..{max_year})")
    print("NULL destination_id in fact:", null_dest)
    print("NULL date_id in fact:", null_date)

if __name__ == "__main__":
    main()
