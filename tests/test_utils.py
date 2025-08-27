#!/usr/bin/env python3
"""
Tests for Utils Package

Comprehensive tests for the utility functions in src/utils/ package.
Tests both databricks_env and sql_utils modules with proper mocking.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.databricks_env import (
    setup_databricks_environment,
    setup_notebook_env,
    setup_local_env,
    cleanup_environment,
    get_environment_info
)

from utils.sql_utils import (
    execute_sql,
    execute_sql_with_result,
    format_sql_multiline,
    validate_catalog_schema_names
)


class TestDatabricksEnvironment:
    """Test Databricks environment setup functions."""
    
    def test_setup_databricks_environment_notebook(self):
        """Test environment detection and setup for notebook environment."""
        with patch.dict(os.environ, {'DATABRICKS_RUNTIME_VERSION': '13.3.x-scala2.12'}):
            with patch('utils.databricks_env.setup_notebook_env') as mock_setup:
                mock_setup.return_value = {
                    'environment': 'notebook',
                    'spark': Mock(),
                    'workspace_client': Mock(),
                    'catalog': 'test_catalog',
                    'schema': 'test_schema'
                }
                
                result = setup_databricks_environment()
                
                assert result['environment'] == 'notebook'
                mock_setup.assert_called_once()
    
    def test_setup_databricks_environment_local(self):
        """Test environment detection and setup for local IDE environment."""
        # Ensure notebook environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            with patch('utils.databricks_env.setup_local_env') as mock_setup:
                mock_setup.return_value = {
                    'environment': 'local',
                    'spark': Mock(),
                    'workspace_client': Mock(),
                    'catalog': 'test_catalog',
                    'schema': 'test_schema'
                }
                
                result = setup_databricks_environment()
                
                assert result['environment'] == 'local'
                mock_setup.assert_called_once()
    
    @patch('utils.databricks_env.WorkspaceClient')
    def test_setup_notebook_env(self, mock_workspace_client):
        """Test notebook environment setup."""
        mock_spark = Mock()
        mock_workspace_client.return_value = Mock()
        
        with patch.dict(os.environ, {
            'UC_DEFAULT_CATALOG': 'test_catalog',
            'UC_DEFAULT_SCHEMA': 'test_schema'
        }):
            # Mock global spark variable
            with patch('utils.databricks_env.spark', mock_spark, create=True):
                result = setup_notebook_env()
                
                assert result['environment'] == 'notebook'
                assert result['spark'] is mock_spark
                assert result['catalog'] == 'test_catalog'
                assert result['schema'] == 'test_schema'
                mock_workspace_client.assert_called_once()
    
    @patch('utils.databricks_env.load_dotenv')
    @patch('utils.databricks_env.DatabricksSession')
    @patch('utils.databricks_env.WorkspaceClient')
    @patch('utils.databricks_env.mlflow')
    def test_setup_local_env_success(self, mock_mlflow, mock_workspace_client, 
                                     mock_databricks_session, mock_load_dotenv):
        """Test successful local environment setup."""
        mock_spark = Mock()
        mock_builder = Mock()
        mock_builder.profile.return_value = mock_builder
        mock_builder.serverless.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        mock_databricks_session.builder = mock_builder
        
        with patch.dict(os.environ, {
            'DATABRICKS_CONFIG_PROFILE': 'test-profile',
            'UC_DEFAULT_CATALOG': 'test_catalog',
            'UC_DEFAULT_SCHEMA': 'test_schema'
        }):
            result = setup_local_env()
            
            assert result['environment'] == 'local'
            assert result['spark'] is mock_spark
            assert result['catalog'] == 'test_catalog'
            assert result['schema'] == 'test_schema'
            
            # Verify Spark session configuration
            mock_spark.sql.assert_called_once_with("USE CATALOG test_catalog")
            
            # Verify MLflow configuration
            mock_mlflow.set_tracking_uri.assert_called_once_with("databricks")
            mock_mlflow.set_registry_uri.assert_called_once_with("databricks-uc")
    
    @patch('utils.databricks_env.load_dotenv')
    @patch('utils.databricks_env.DatabricksSession')
    def test_setup_local_env_failure(self, mock_databricks_session, mock_load_dotenv):
        """Test local environment setup failure (dry-run mode)."""
        mock_databricks_session.builder.profile.side_effect = Exception("Connection failed")
        
        with patch.dict(os.environ, {
            'UC_DEFAULT_CATALOG': 'test_catalog',
            'UC_DEFAULT_SCHEMA': 'test_schema'
        }):
            result = setup_local_env()
            
            assert result['environment'] == 'local'
            assert result['spark'] is None  # Dry-run mode
            assert result['workspace_client'] is None
            assert result['catalog'] == 'test_catalog'
            assert result['schema'] == 'test_schema'
    
    def test_cleanup_environment_local_with_spark(self):
        """Test cleanup for local environment with active Spark session."""
        mock_spark = Mock()
        config = {
            'environment': 'local',
            'spark': mock_spark
        }
        
        cleanup_environment(config)
        mock_spark.stop.assert_called_once()
    
    def test_cleanup_environment_notebook(self):
        """Test cleanup for notebook environment (should do nothing)."""
        mock_spark = Mock()
        config = {
            'environment': 'notebook',
            'spark': mock_spark
        }
        
        cleanup_environment(config)
        mock_spark.stop.assert_not_called()
    
    def test_cleanup_environment_no_spark(self):
        """Test cleanup when no Spark session exists."""
        config = {
            'environment': 'local',
            'spark': None
        }
        
        # Should not raise exception
        cleanup_environment(config)
    
    def test_cleanup_environment_spark_stop_error(self):
        """Test cleanup when Spark stop raises exception."""
        mock_spark = Mock()
        mock_spark.stop.side_effect = Exception("Stop failed")
        config = {
            'environment': 'local',
            'spark': mock_spark
        }
        
        # Should not raise exception
        cleanup_environment(config)
    
    def test_get_environment_info_local(self):
        """Test getting environment info for local setup."""
        config = {
            'environment': 'local',
            'catalog': 'test_catalog',
            'schema': 'test_schema',
            'spark': Mock()
        }
        
        with patch.dict(os.environ, {
            'DATABRICKS_CONFIG_PROFILE': 'test-profile',
            'DATABRICKS_HOST': 'https://test.databricks.com'
        }):
            info = get_environment_info(config)
            
            assert info['environment'] == 'local'
            assert info['catalog'] == 'test_catalog'
            assert info['schema'] == 'test_schema'
            assert info['spark_available'] == 'Yes'
            assert info['profile'] == 'test-profile'
            assert info['host'] == 'https://test.databricks.com'
    
    def test_get_environment_info_notebook_dry_run(self):
        """Test getting environment info for notebook with no Spark."""
        config = {
            'environment': 'notebook',
            'catalog': 'test_catalog',
            'schema': 'test_schema',
            'spark': None
        }
        
        info = get_environment_info(config)
        
        assert info['environment'] == 'notebook'
        assert info['spark_available'] == 'No (dry-run mode)'
        assert 'profile' not in info  # Only for local env


class TestSQLUtils:
    """Test SQL utility functions."""
    
    def test_execute_sql_success_with_spark(self):
        """Test successful SQL execution with Spark session."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_result.collect.return_value = []
        mock_spark.sql.return_value = mock_result
        
        result = execute_sql(mock_spark, "CREATE TABLE test", "Creating test table")
        
        assert result is True
        mock_spark.sql.assert_called_once_with("CREATE TABLE test")
        mock_result.collect.assert_called_once()
    
    def test_execute_sql_success_without_collect(self):
        """Test SQL execution where result doesn't have collect method."""
        mock_spark = Mock()
        mock_result = Mock()
        del mock_result.collect  # Remove collect method
        mock_spark.sql.return_value = mock_result
        
        result = execute_sql(mock_spark, "USE CATALOG test", "Using catalog")
        
        assert result is True
        mock_spark.sql.assert_called_once_with("USE CATALOG test")
    
    def test_execute_sql_dry_run_mode(self):
        """Test SQL execution in dry-run mode (no Spark session)."""
        result = execute_sql(None, "CREATE TABLE test", "Creating test table")
        
        assert result is True
        # Should succeed without actual execution
    
    def test_execute_sql_already_exists_error(self):
        """Test SQL execution with 'already exists' error."""
        mock_spark = Mock()
        mock_spark.sql.side_effect = Exception("Table already exists")
        
        result = execute_sql(mock_spark, "CREATE TABLE test", "Creating test table")
        
        assert result is True  # Should return True for already exists
    
    def test_execute_sql_other_error(self):
        """Test SQL execution with other types of errors."""
        mock_spark = Mock()
        mock_spark.sql.side_effect = Exception("Permission denied")
        
        result = execute_sql(mock_spark, "CREATE TABLE test", "Creating test table")
        
        assert result is False
    
    def test_execute_sql_with_result_success(self):
        """Test execute_sql_with_result success case."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_spark.sql.return_value = mock_result
        
        success, result = execute_sql_with_result(mock_spark, "SHOW TABLES", "Showing tables")
        
        assert success is True
        assert result is mock_result
        mock_spark.sql.assert_called_once_with("SHOW TABLES")
    
    def test_execute_sql_with_result_dry_run(self):
        """Test execute_sql_with_result in dry-run mode."""
        success, result = execute_sql_with_result(None, "SHOW TABLES", "Showing tables")
        
        assert success is True
        assert result is None
    
    def test_execute_sql_with_result_error(self):
        """Test execute_sql_with_result with error."""
        mock_spark = Mock()
        mock_spark.sql.side_effect = Exception("Query failed")
        
        success, result = execute_sql_with_result(mock_spark, "SHOW TABLES", "Showing tables")
        
        assert success is False
        assert result is None
    
    def test_format_sql_multiline(self):
        """Test SQL formatting for multiline statements."""
        sql = """
        CREATE TABLE test (
            id INT,
            name STRING
        )
        """
        
        formatted = format_sql_multiline(sql, indent=2)
        
        lines = formatted.split('\n')
        assert all(line.startswith('  ') for line in lines if line.strip())
        assert 'CREATE TABLE test' in formatted
        assert 'id INT' in formatted
    
    def test_validate_catalog_schema_names_valid(self):
        """Test validation of valid catalog and schema names."""
        is_valid, error = validate_catalog_schema_names("test_catalog", "test_schema")
        
        assert is_valid is True
        assert error == ""
    
    def test_validate_catalog_schema_names_valid_underscore_start(self):
        """Test validation with names starting with underscore."""
        is_valid, error = validate_catalog_schema_names("_test", "_schema")
        
        assert is_valid is True
        assert error == ""
    
    def test_validate_catalog_schema_names_invalid_catalog_number_start(self):
        """Test validation with catalog name starting with number."""
        is_valid, error = validate_catalog_schema_names("1test", "schema")
        
        assert is_valid is False
        assert "Invalid catalog name" in error
        assert "must start with letter/underscore" in error
    
    def test_validate_catalog_schema_names_invalid_schema_special_chars(self):
        """Test validation with schema name containing special characters."""
        is_valid, error = validate_catalog_schema_names("test", "schema-name")
        
        assert is_valid is False
        assert "Invalid schema name" in error
    
    def test_validate_catalog_schema_names_too_long(self):
        """Test validation with names that are too long."""
        long_name = "a" * 256  # Longer than 255 character limit
        
        is_valid, error = validate_catalog_schema_names(long_name, "schema")
        
        assert is_valid is False
        assert "too long" in error
    
    def test_validate_catalog_schema_names_no_schema(self):
        """Test validation with only catalog name."""
        is_valid, error = validate_catalog_schema_names("test_catalog")
        
        assert is_valid is True
        assert error == ""


class TestUtilsIntegration:
    """Integration tests for utils package."""
    
    def test_utils_package_imports(self):
        """Test that utils package imports work correctly."""
        # Test direct imports
        from utils import (
            setup_databricks_environment,
            setup_notebook_env,
            setup_local_env,
            cleanup_environment,
            execute_sql
        )
        
        # Verify functions are callable
        assert callable(setup_databricks_environment)
        assert callable(setup_notebook_env)
        assert callable(setup_local_env)
        assert callable(cleanup_environment)
        assert callable(execute_sql)
    
    def test_utils_submodule_imports(self):
        """Test that submodule imports work correctly."""
        # Test submodule imports
        from utils.databricks_env import setup_databricks_environment
        from utils.sql_utils import execute_sql
        
        assert callable(setup_databricks_environment)
        assert callable(execute_sql)
    
    @patch('utils.databricks_env.DatabricksSession')
    @patch('utils.databricks_env.load_dotenv')
    def test_environment_and_sql_integration(self, mock_load_dotenv, mock_databricks_session):
        """Test integration between environment setup and SQL utils."""
        # Mock successful environment setup
        mock_spark = Mock()
        mock_builder = Mock()
        mock_builder.profile.return_value = mock_builder
        mock_builder.serverless.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        mock_databricks_session.builder = mock_builder
        
        # Setup environment
        config = setup_local_env()
        
        # Use SQL utils with the configured environment
        result = execute_sql(config['spark'], "CREATE TABLE test", "Test creation")
        
        assert result is True
        mock_spark.sql.assert_called_with("CREATE TABLE test")
        
        # Test cleanup
        cleanup_environment(config)
        mock_spark.stop.assert_called_once()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])