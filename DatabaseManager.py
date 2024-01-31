import psycopg2

class DatabaseManager:
    def __init__(self, connection_params):
        self.connection_params = connection_params
        self.connection = None

    def connect_to_database(self):
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            print('Database connection established.')
        except Exception as e:
            print(f'Error connecting to the database: {e}')
            raise e

    def get_locations(self):
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT * FROM locations LIMIT 2')
            try:
                results = cursor.fetchall()
                print(f'{len(results)} locations fetched from the database.')
                return results
            except Exception as e:
                print(f'Error fetching locations: {e}')
                raise e

    def store_criteria(self, location_id, criteria):
        with self.connection.cursor() as cursor:
            cursor.execute('UPDATE locations SET criteria = %s WHERE id = %s', (criteria, location_id))
            try:
                self.connection.commit()
                print(f'Criteria for location {location_id} stored in the database.')
            except Exception as e:
                print(f'Error storing criteria: {e}')
                self.connection.rollback()
                raise e
