#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.build_world_bank_tourism import main as build_wb
from src.etl.load_dim_destination import main as load_dest
from src.etl.load_fact_flow import main as load_fact
from src.etl.seed_dim_date import main as seed_date
from src.etl.update_dim_destination_country_code import main as update_codes
from src.utils.logging_utils import configure_logging, get_logger

configure_logging()
logger = get_logger("etl.load_all_data")


def main():
    logger.info("=" * 60)
    logger.info("Загрузка данных в базу данных")
    logger.info("=" * 60)

    try:
        logger.info("1/5 Загрузка dim_date")
        seed_date()

        logger.info("2/5 Подготовка данных World Bank")
        build_wb()

        logger.info("3/5 Загрузка dim_destination")
        load_dest()

        logger.info("4/5 Обновление country_code")
        update_codes()

        logger.info("5/5 Загрузка fact_flow")
        load_fact()

        logger.info("=" * 60)
        logger.info("Готово! Все данные загружены.")
        logger.info("=" * 60)
    except Exception:
        logger.exception("Ошибка при загрузке данных")
        raise


if __name__ == "__main__":
    main()
