import csv
import json
import os

input_file = os.path.join(os.path.dirname(__file__), "financial_transactions.csv")
output_file = os.path.join(os.path.dirname(__file__), "nebula_queries.json")

TAG_NAME = "Person"
EDGE_NAME = "Transaction"

def generate_ngql(data):
    queries = []
    inserted_vertices = set()

    for record in data:
        source, amount, target = record

        # Insert source vertex if not already inserted
        if source not in inserted_vertices:
            queries.append(
                f'INSERT VERTEX {TAG_NAME}() VALUES "{source}":();'
            )
            inserted_vertices.add(source)

        # Insert target vertex if not already inserted
        if target not in inserted_vertices:
            queries.append(
                f'INSERT VERTEX {TAG_NAME}() VALUES "{target}":();'
            )
            inserted_vertices.add(target)

        # Insert edge
        queries.append(
            f'INSERT EDGE {EDGE_NAME}(amount) VALUES "{source}"->"{target}":({amount});'
        )

    return queries


def main():
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = [(row["borrower_name"], row["amount"], row["lender_name"]) for row in reader]

    queries = generate_ngql(data)

    # Output as a JSON list (ordered)
    with open(output_file, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"✅ Queries written to {output_file}")


if __name__ == "__main__":
    main()