import os
from typing import List, Any
from src.task_1_ import run_task_one
from src.task_2_ import run_task_two


def main(data_set_path: str, train_set_1: str, date_list: List[Any]):
    try:
        run_task_one(data_set_path, train_set_1)
    except Exception:
        pass
    try:
        run_task_two(data_set_path, date_list)
    except Exception:
        pass


if __name__ == '__main__':
    main("../IML_Hackathon/data_src/agoda_cancellation_train.csv",
         "../IML_Hackathon/data_src/Agoda_Test_1.csv",
         ["2023-06-07", "2023-06-08", "2023-06-09"])
