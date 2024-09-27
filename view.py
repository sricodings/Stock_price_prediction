import sqlite3

def view_table_data(table_name):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute(f"SELECT * FROM {table_name}")
        rows = c.fetchall()
        if rows:
            columns = [description[0] for description in c.description]
            print(f"Table: {table_name}")
            print(" | ".join(columns))
            print("-" * (len(" | ".join(columns))))
            for row in rows:
                print(" | ".join(map(str, row)))
        else:
            print(f"No data found in table {table_name}.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
if __name__ == "__main__":
    table_name = input("Enter the table name to view data: ")
    view_table_data(table_name)
