import os
import time


def check_local_index_validity(index_name: str):
    # Checks the local index, if not exists or is modification timestamp older than predefined period, download latest version from storage
    if not os.path.exists(index_name):
        download_index_from_storage(index_name)
    time_index_last_modified = os.path.getmtime(index_name)
    current_time = time.time()
    max_index_age = 24 * 60 * 60  # 24 hours
    if current_time - time_index_last_modified > max_index_age:
        download_index_from_storage(index_name)
    return None


def download_index_from_storage(index_name: str):
    # TODO Not implemented
    raise NotImplementedError
