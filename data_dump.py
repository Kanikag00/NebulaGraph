import os
import json
import argparse
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import time

def load_ngql_commands(filepath):
    with open(filepath, 'r') as json_file:
        return json.load(json_file)

def main():
    parser = argparse.ArgumentParser(description="Load NGQL commands into Nebula Graph")
    parser.add_argument("input_file", help="Path to JSON file containing NGQL commands")
    args = parser.parse_args()

    host = os.getenv("NEBULA_HOST", "10.2.2.22")
    port = int(os.getenv("NEBULA_PORT", "9669"))
    user = os.getenv("NEBULA_USER", "root")
    password = os.getenv("NEBULA_PASSWORD", "nebula")
    space = os.getenv("NEBULA_SPACE", "financial")

    config = Config()
    config.max_connection_pool_size = 10

    # Init connection pool
    connection_pool = ConnectionPool()
    ok = connection_pool.init([(host, port)], config)

    if not ok:
        print("Connection to the server failed.")
        return

    # Get session from the pool
    session = connection_pool.get_session(user, password)

    # Select space
    session.execute(f"USE {space};")

    # Load NGQL commands from JSON file
    ngql_commands = load_ngql_commands(args.input_file)

    # Execute each NGQL command
    for command in ngql_commands:
        print(f"Executing command: {command}")
        result = session.execute(command)
        time.sleep(2)
        print(f"Result: {result}")

    # Release session
    session.release()
    # Close the pool
    connection_pool.close()

if __name__ == "__main__":
    main()