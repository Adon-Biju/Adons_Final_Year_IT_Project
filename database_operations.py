import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime


DB_CONFIG = {
    'dbname': 'facial_recognition_data',
    'user': 'postgres',
    'password': 'master',
    'host': 'localhost',
    'port': '5050'
}

def create_database():
    """Create database if it doesn't exist"""
    temp_config = DB_CONFIG.copy()
    temp_config['dbname'] = 'postgres'
    
    try:
        conn = psycopg2.connect(**temp_config)
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", 
                   (DB_CONFIG['dbname'],))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            print(f"Database '{DB_CONFIG['dbname']}' created successfully")
    except Exception as e:
        print(f"Error checking/creating database: {str(e)}")
    finally:
        cur.close()
        conn.close()

def execute_query(query, params=None, fetch_all=False):
    """Execute a database query and return results"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)
        try:
            cur.execute(query, params or ())
            conn.commit()
            if fetch_all:
                return cur.fetchall()
            result = cur.fetchone()
            
            return result
        except psycopg2.OperationalError as e:
            if "does not exist" in str(e):
                create_database()
                return execute_query(query, params, fetch_all)
            print(f"Database error: {str(e)}")  
            return None
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        if "no results to fetch" not in str(e):
            print(f"Database error: {str(e)}")
        return None


TABLES = {
    'models': """
        CREATE TABLE IF NOT EXISTS models (
            model_id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) UNIQUE NOT NULL
        )
    """,
    'people': """
        CREATE TABLE IF NOT EXISTS people (
            person_id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL
        )
    """,
    'recognition_tests': """
        CREATE TABLE IF NOT EXISTS recognition_tests (
            test_id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(model_id),
            person_id INTEGER REFERENCES people(person_id),
            test_timestamp TIMESTAMP NOT NULL,
            total_attempts INTEGER NOT NULL,
            successful_recognitions INTEGER NOT NULL,
            avg_confidence DECIMAL(5,4) NOT NULL,
            avg_processing_time DECIMAL(6,3) NOT NULL,
            avg_recognition_rate DECIMAL(5,2) NOT NULL
        )
    """,
    'model_aggregate_stats': """
        CREATE TABLE IF NOT EXISTS model_aggregate_stats (
            model_id INTEGER REFERENCES models(model_id) PRIMARY KEY,
            calculation_timestamp TIMESTAMP NOT NULL,
            total_tests INTEGER NOT NULL,
            overall_recognition_rate DECIMAL(5,2),
            overall_processing_time DECIMAL(6,3),
            overall_confidence DECIMAL(5,2)
        )
    """,
    'failed_tests': """
        CREATE TABLE IF NOT EXISTS failed_tests (
            model_id INTEGER REFERENCES models(model_id) PRIMARY KEY,
            count INTEGER DEFAULT 0,
            last_updated TIMESTAMP NOT NULL
        )
    """
}

def init_database():
    """Initialize all database tables and default data"""
    create_database()
    try:
        
        for table_name, create_statement in TABLES.items():
            execute_query(create_statement)
        
        
        execute_query("""
            INSERT INTO models (model_name)
            VALUES ('ArcFace'), ('Facenet'), ('Dlib')
            ON CONFLICT (model_name) DO NOTHING
        """)
        print("Database initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

def get_or_create_person(name):
    """Get existing person or create new one"""
    result = execute_query(
        "SELECT person_id FROM people WHERE name = %s",
        (name,)
    )
    if result:
        return result[0]
    
    result = execute_query(
        "INSERT INTO people (name) VALUES (%s) RETURNING person_id",
        (name,)
    )
    return result[0]

def save_test_results(model_name, person_name, stats):
    """Save individual test results"""
    try:
        
        result = execute_query(
            "SELECT model_id FROM models WHERE model_name = %s",
            (model_name,)
        )
        if not result:
            return False
            
        model_id = result['model_id']
        person_id = get_or_create_person(person_name)
        
        # Save test results
        execute_query("""
            INSERT INTO recognition_tests (
                model_id, person_id, test_timestamp,
                total_attempts, successful_recognitions,
                avg_confidence, avg_processing_time, avg_recognition_rate
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_id, person_id, datetime.now(),
            stats['total_attempts'], stats['successful_recognitions'],
            stats['avg_confidence'], stats['avg_time'],
            stats['avg_rate']
        ))
        return True
    except Exception as e:
        print(f"Error saving test results: {str(e)}")
        return False

def get_model_stats():
    """Get current model statistics"""
    return execute_query("""
        SELECT 
            m.model_name,
            COUNT(*) as total_tests,
            AVG(rt.avg_recognition_rate) as avg_recognition_rate,
            AVG(rt.avg_processing_time) as avg_processing_time,
            AVG(rt.avg_confidence) as avg_confidence
        FROM models m
        JOIN recognition_tests rt ON m.model_id = rt.model_id
        GROUP BY m.model_name
        ORDER BY m.model_name
    """, fetch_all=True)

def save_aggregate_stats():
    """Update aggregate statistics for all models"""
    try:
        stats = get_model_stats()
        if not stats:
            return False
            
        for stat in stats:
            execute_query("""
                INSERT INTO model_aggregate_stats (
                    model_id, calculation_timestamp, total_tests,
                    overall_recognition_rate, overall_processing_time, overall_confidence
                )
                SELECT 
                    m.model_id, CURRENT_TIMESTAMP, %s, %s, %s, %s
                FROM models m
                WHERE m.model_name = %s
                ON CONFLICT (model_id) DO UPDATE SET
                    calculation_timestamp = EXCLUDED.calculation_timestamp,
                    total_tests = EXCLUDED.total_tests,
                    overall_recognition_rate = EXCLUDED.overall_recognition_rate,
                    overall_processing_time = EXCLUDED.overall_processing_time,
                    overall_confidence = EXCLUDED.overall_confidence
            """, (
                stat['total_tests'],
                stat['avg_recognition_rate'],
                stat['avg_processing_time'],
                stat['avg_confidence'],
                stat['model_name']
            ))
        return True
    except Exception as e:
        print(f"Error saving aggregate stats: {str(e)}")
        return False
    
def record_failed_tests(model_name):
    """Record an failed attempt for a model"""
    try:
        
        result = execute_query(
            "SELECT model_id FROM models WHERE model_name = %s",
            (model_name,)
        )
        if not result:
            return False
            
        model_id = result['model_id']
        
       
        execute_query("""
            INSERT INTO failed_tests (model_id, count, last_updated)
            VALUES (%s, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (model_id) DO UPDATE SET
                count = failed_tests.count + 1,
                last_updated = CURRENT_TIMESTAMP
        """, (model_id,))
        return True
    except Exception as e:
        print(f"Error recording failed tests: {str(e)}")
        return False

def get_failed_tests_stats():
    """Get failed tests counts for all models"""
    return execute_query("""
        SELECT 
            m.model_name,
            COALESCE(ur.count, 0) as fail_count,
            ur.last_updated
        FROM models m
        LEFT JOIN failed_tests ur ON m.model_id = ur.model_id
        ORDER BY m.model_name
    """, fetch_all=True)

def get_historical_aggregate_stats():
    """Get latest statistics for each model"""
    return execute_query("""
        SELECT 
            m.model_name,
            mas.total_tests,
            mas.overall_recognition_rate,
            mas.overall_processing_time,
            mas.overall_confidence
        FROM model_aggregate_stats mas
        JOIN models m ON mas.model_id = m.model_id
        ORDER BY m.model_name
    """, fetch_all=True)