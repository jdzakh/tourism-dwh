from src.etl.build_world_bank_tourism import main as build_wb
from src.etl.load_dim_destination import main as load_dest
from src.etl.update_dim_destination_country_code import main as update_codes
from src.etl.load_fact_flow import main as load_fact


def main():
    build_wb()
    load_dest()
    update_codes()
    load_fact()


if __name__ == "__main__":
    main()
