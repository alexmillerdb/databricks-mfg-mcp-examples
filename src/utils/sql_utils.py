#!/usr/bin/env python3
"""
SQL Utilities for Databricks Operations

Common functions for executing SQL statements with proper error handling,
logging, and dry-run support.
"""

from typing import Optional, Any


def execute_sql(spark: Optional[Any], sql: str, description: Optional[str] = None) -> bool:
    """
    Execute SQL statement with error handling and logging.
    
    Provides consistent error handling for "already exists" conditions and
    supports dry-run mode when spark session is not available.
    
    Args:
        spark: SparkSession object or None for dry-run mode
        sql: SQL statement to execute
        description: Optional description for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if description:
            print(f"📋 {description}")
        
        if spark:
            result = spark.sql(sql)
            # For CREATE/GRANT statements that don't return data
            if result is not None and hasattr(result, 'collect'):
                result.collect()
        else:
            # Dry-run mode - just print the SQL
            print(f"   SQL (dry-run): {sql[:200]}{'...' if len(sql) > 200 else ''}")
        
        print("   ✅ Success")
        return True
        
    except Exception as e:
        error_msg = str(e)
        # Check if it's an "already exists" error which is OK
        if "already exists" in error_msg.lower():
            print("   ℹ️  Already exists (skipping)")
            return True
        else:
            print(f"   ❌ Error: {error_msg}")
            return False


def execute_sql_with_result(spark: Optional[Any], sql: str, description: Optional[str] = None) -> tuple[bool, Optional[Any]]:
    """
    Execute SQL statement and return both success status and result.
    
    Useful for queries that need to return data (like SHOW TABLES).
    
    Args:
        spark: SparkSession object or None for dry-run mode
        sql: SQL statement to execute
        description: Optional description for logging
        
    Returns:
        Tuple of (success: bool, result: Any or None)
    """
    try:
        if description:
            print(f"📋 {description}")
        
        if spark:
            result = spark.sql(sql)
            print("   ✅ Success")
            return True, result
        else:
            # Dry-run mode - just print the SQL
            print(f"   SQL (dry-run): {sql[:200]}{'...' if len(sql) > 200 else ''}")
            print("   ✅ Success (dry-run)")
            return True, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Error: {error_msg}")
        return False, None


def format_sql_multiline(sql: str, indent: int = 4) -> str:
    """
    Format multiline SQL for better logging in dry-run mode.
    
    Args:
        sql: SQL statement to format
        indent: Number of spaces to indent each line
        
    Returns:
        Formatted SQL string
    """
    lines = sql.strip().split('\n')
    indented_lines = [' ' * indent + line.strip() for line in lines if line.strip()]
    return '\n'.join(indented_lines)


def validate_catalog_schema_names(catalog: str, schema: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate catalog and schema names according to Unity Catalog naming rules.
    
    Args:
        catalog: Catalog name to validate
        schema: Optional schema name to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    import re
    
    # Unity Catalog naming pattern: alphanumeric and underscores, cannot start with number
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    
    if not re.match(pattern, catalog):
        return False, f"Invalid catalog name '{catalog}': must start with letter/underscore and contain only letters, numbers, underscores"
    
    if schema and not re.match(pattern, schema):
        return False, f"Invalid schema name '{schema}': must start with letter/underscore and contain only letters, numbers, underscores"
    
    # Check length limits (Unity Catalog limits)
    if len(catalog) > 255:
        return False, f"Catalog name '{catalog}' is too long (max 255 characters)"
    
    if schema and len(schema) > 255:
        return False, f"Schema name '{schema}' is too long (max 255 characters)"
    
    return True, ""