import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="ddos_attack",
        user="postgres",
        password="kendz2k3",
        host="localhost",
        port="5432"
    )
