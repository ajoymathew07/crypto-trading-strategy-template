import pandas as pd
from untrade.client import Client
import sys
import os 
import uuid
def perform_backtest(csv_file_path):
    client = Client()
    result = client.backtest(
        # jupyter_id="bytefoot",  # the one you use to login to jupyter.untrade.io
        jupyter_id="ee23b181",  # the one you use to login to jupyter.untrade.io
        # jupyter_id="ep23b027",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,  # Adjust leverage as needed
        # result_type='Q',
    )
    return result

def perform_backtest_large_csv(csv_file_path):
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024  # Setting chunk size to 90 MB
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size

    if total_size <= chunk_size:
        # For smaller files, perform a normal backtest
        result = client.backtest(file_path=csv_file_path, leverage=1, jupyter_id="test")
        for value in result:
            print(value)
        return result

    chunk_number = 0
    with open(csv_file_path, "rb") as f:
        while chunk_data := f.read(chunk_size):
            chunk_file_path = f"/tmp/{file_id}chunk{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            result = client.backtest(file_path=chunk_file_path, leverage=1, jupyter_id="test", file_id=file_id,
                                     chunk_number=chunk_number, total_chunks=total_chunks)
            for value in result:
                print(value)

            os.remove(chunk_file_path)
            chunk_number += 1

    return result

def format_print(result):
    print("Static Statistics:")
    for k, v in result['result']['static_statistics'].items():
        print(f" |- {k}:  {v}")
    
    print("\n\nCompound Statistics:")
    for k, v in result['result']['compound_statistics'].items():
        print(f" |- {k}:  {v}")

def format_print_q(result):
    for item in result['result']:
        print(f"{item['index']}: {item['Benchmark Beaten?']}")

if __name__ == "__main__":
    # Perform backtest on processed data
    csv_file_path = sys.argv[1]
    backtest_result = perform_backtest(csv_file_path)
    # print(backtest_result)
    last_value = ""
    for value in backtest_result:
        last_value += value
    last_value = last_value.replace("data: ", "")
    try:
        format_print(eval(last_value, {'Timestamp': pd.Timestamp}))
    except Exception as e:
        print(last_value)

        print("\n\n")
        format_print_q(eval(last_value, {'Timestamp': pd.Timestamp}))
    
