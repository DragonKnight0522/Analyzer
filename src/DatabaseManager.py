import psycopg2
import os


class DatabaseManager:
    def __init__(self):
        print()
        self.connection_params = {
            'host': os.environ["DATABASE_HOST"],
            'database': os.environ["DATABASE_NAME"],
            'user': os.environ["DATABASE_USER"],
            'password': os.environ["DATABASE_PASSWORD"],
            'port': os.environ["DATABASE_PORT"]
        }
        self.connection = None

    def connect_to_database(self):
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise e

    def get_locations(self):
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT locations.id, yelp_feature, yelp_amenities, yelp_about, yelp_menu, yelp_name, yelp_reviews, crawl.data
                    FROM locations 
                    LEFT JOIN scrap ON scrap.location_id = locations.id
                    LEFT JOIN crawl ON crawl.location_id = locations.id
                    WHERE is_analyzed IS NOT TRUE AND (scrap.location_id IS NOT NULL OR crawl.location_id IS NOT NULL)
                    LIMIT 1;
                """
            )
            try:
                results = cursor.fetchall()
                print(f"{len(results)} locations fetched from the database.")
                return results
            except Exception as e:
                print(f"Error fetching locations: {e}")
                raise e

    def store_criteria(self, location_id, criteria):
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE locations SET criteria = %s WHERE id = %s",
                (criteria, location_id),
            )
            try:
                self.connection.commit()
                print(
                    f"Criteria for location {location_id} stored in the database.")
            except Exception as e:
                print(f"Error storing criteria: {e}")
                self.connection.rollback()
                raise e
