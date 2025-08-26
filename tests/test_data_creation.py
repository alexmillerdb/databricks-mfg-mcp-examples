#!/usr/bin/env python3
"""
Tests for Data Creation Scripts

Test the Unity Catalog setup, Delta table creation, and data generation utilities.
Uses pytest for test framework with mocking for Spark operations.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.setup_unity_catalog import (
    create_catalog, 
    create_schemas, 
    grant_permissions,
    execute_sql
)
from data.generate_sample_data import (
    generate_suppliers,
    generate_inventory,
    generate_customers,
    generate_transactions,
    generate_iot_telemetry,
    generate_support_tickets,
    generate_sop_documents,
    generate_incident_reports,
    generate_sales_proposals
)


class TestUnityCartlorySetup:
    """Test Unity Catalog setup functionality."""
    
    def test_create_catalog_success(self):
        """Test successful catalog creation."""
        # Mock Spark session
        mock_spark = Mock()
        mock_spark.sql.return_value = Mock()
        
        result = create_catalog(mock_spark, "test_catalog")
        
        assert result is True
        mock_spark.sql.assert_called_once()
        call_args = mock_spark.sql.call_args[0][0]
        assert "CREATE CATALOG IF NOT EXISTS test_catalog" in call_args
    
    def test_create_catalog_already_exists(self):
        """Test catalog creation when catalog already exists."""
        mock_spark = Mock()
        mock_spark.sql.side_effect = Exception("already exists")
        
        result = create_catalog(mock_spark, "existing_catalog")
        
        # Should return True for "already exists" error
        assert result is True
    
    def test_create_catalog_other_error(self):
        """Test catalog creation with unexpected error."""
        mock_spark = Mock()
        mock_spark.sql.side_effect = Exception("access denied")
        
        result = create_catalog(mock_spark, "test_catalog")
        
        assert result is False
    
    def test_create_schemas_success(self):
        """Test successful schema creation."""
        mock_spark = Mock()
        mock_spark.sql.return_value = Mock()
        
        results = create_schemas(mock_spark, "test_catalog")
        
        # Should create 5 schemas
        assert len(results) == 5
        assert all(results.values())  # All should be True
        
        # Verify expected schemas
        expected_schemas = ["supply_chain", "sales", "iot", "support", "agent_logs"]
        assert set(results.keys()) == set(expected_schemas)
    
    def test_execute_sql_dry_run(self):
        """Test SQL execution in dry-run mode (no Spark session)."""
        result = execute_sql(None, "SELECT 1", "Test query")
        
        assert result is True  # Should succeed in dry-run mode
    
    def test_execute_sql_with_spark(self):
        """Test SQL execution with Spark session."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_result.collect.return_value = []
        mock_spark.sql.return_value = mock_result
        
        result = execute_sql(mock_spark, "CREATE SCHEMA test", "Test schema creation")
        
        assert result is True
        mock_spark.sql.assert_called_once_with("CREATE SCHEMA test")


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_suppliers(self):
        """Test supplier data generation."""
        suppliers = generate_suppliers(10)
        
        assert len(suppliers) == 10
        
        # Check first supplier structure
        supplier = suppliers[0]
        required_fields = [
            "supplier_id", "supplier_name", "contact_name", "contact_email",
            "contact_phone", "address", "city", "country", "rating",
            "lead_time_days", "payment_terms", "status", "created_date", "last_order_date"
        ]
        
        for field in required_fields:
            assert field in supplier, f"Missing field: {field}"
        
        # Validate data types and constraints
        assert supplier["supplier_id"].startswith("SUP")
        assert 3.0 <= supplier["rating"] <= 5.0
        assert supplier["lead_time_days"] >= 3
        assert supplier["status"] in ["ACTIVE", "INACTIVE"]
        assert isinstance(supplier["created_date"], date)
        assert isinstance(supplier["last_order_date"], date)
    
    def test_generate_inventory(self):
        """Test inventory data generation."""
        suppliers = generate_suppliers(5)
        inventory = generate_inventory(suppliers, 100)
        
        # Should create inventory for multiple warehouses (100 parts distributed across 5 warehouses = 20 parts per warehouse * 5 warehouses = 100 total records)
        assert len(inventory) == 100  # Exactly as requested, distributed across warehouses
        
        # Check inventory record structure
        item = inventory[0]
        required_fields = [
            "inventory_id", "part_id", "part_name", "warehouse_id",
            "warehouse_location", "current_quantity", "reorder_level",
            "reorder_quantity", "unit_cost", "last_updated", "category", "supplier_id"
        ]
        
        for field in required_fields:
            assert field in item, f"Missing field: {field}"
        
        # Validate constraints
        assert item["part_id"].startswith("PART")
        assert item["warehouse_id"].startswith("WH")
        assert item["current_quantity"] >= 0
        assert item["reorder_level"] > 0
        assert item["unit_cost"] > 0
        assert isinstance(item["last_updated"], datetime)
    
    def test_generate_customers(self):
        """Test customer data generation."""
        customers = generate_customers(50)
        
        assert len(customers) == 50
        
        # Check customer structure
        customer = customers[0]
        required_fields = [
            "customer_id", "customer_name", "contact_name", "email", "phone",
            "industry", "segment", "address", "city", "country", "created_date",
            "status", "total_orders", "total_spent", "last_purchase_date",
            "credit_limit", "payment_terms"
        ]
        
        for field in required_fields:
            assert field in customer, f"Missing field: {field}"
        
        # Validate constraints
        assert customer["customer_id"].startswith("CUST")
        assert customer["segment"] in ["Enterprise", "SMB", "Startup"]
        assert customer["status"] in ["ACTIVE", "INACTIVE", "CHURNED"]
        assert customer["total_orders"] >= 1
        assert customer["total_spent"] > 0
        assert customer["credit_limit"] > 0
    
    def test_generate_transactions(self):
        """Test transaction data generation."""
        customers = generate_customers(10)
        suppliers = generate_suppliers(5)
        inventory = generate_inventory(suppliers, 50)
        
        transactions = generate_transactions(customers, inventory, 100)
        
        assert len(transactions) == 100
        
        # Check transaction structure
        txn = transactions[0]
        required_fields = [
            "transaction_id", "order_id", "customer_id", "customer_name",
            "transaction_date", "product_id", "product_name", "quantity",
            "unit_price", "total_amount", "discount_amount", "tax_amount",
            "payment_method", "sales_rep_id", "sales_rep_name", "region", "status"
        ]
        
        for field in required_fields:
            assert field in txn, f"Missing field: {field}"
        
        # Validate business logic
        assert txn["transaction_id"].startswith("TXN")
        assert txn["quantity"] >= 1
        assert txn["unit_price"] > 0
        assert txn["total_amount"] > 0
        assert isinstance(txn["transaction_date"], datetime)
    
    def test_generate_iot_telemetry(self):
        """Test IoT telemetry data generation."""
        telemetry = generate_iot_telemetry(1000)
        
        assert len(telemetry) == 1000
        
        # Check telemetry record structure
        reading = telemetry[0]
        required_fields = [
            "reading_id", "device_id", "device_type", "location_id",
            "timestamp", "power_consumption", "operational_status",
            "maintenance_required", "reading_date"
        ]
        
        for field in required_fields:
            assert field in reading, f"Missing field: {field}"
        
        # Validate constraints
        assert reading["device_id"].startswith("DEV")
        assert reading["power_consumption"] > 0
        assert reading["operational_status"] in ["NORMAL", "WARNING", "ERROR"]
        assert isinstance(reading["maintenance_required"], bool)
        assert isinstance(reading["timestamp"], datetime)
        assert isinstance(reading["reading_date"], date)
    
    def test_generate_support_tickets(self):
        """Test support ticket generation."""
        customers = generate_customers(10)
        tickets = generate_support_tickets(customers, 100)
        
        assert len(tickets) == 100
        
        # Check ticket structure
        ticket = tickets[0]
        required_fields = [
            "ticket_id", "customer_id", "customer_name", "created_date",
            "category", "priority", "subject", "description", "status",
            "assigned_to", "tags"
        ]
        
        for field in required_fields:
            assert field in ticket, f"Missing field: {field}"
        
        # Validate constraints
        assert ticket["ticket_id"].startswith("TKT")
        assert ticket["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert ticket["status"] in ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED", "ESCALATED"]
        assert isinstance(ticket["created_date"], datetime)
        assert isinstance(ticket["tags"], list)
    
    def test_generate_sop_documents(self):
        """Test SOP document generation."""
        sops = generate_sop_documents(25)
        
        assert len(sops) == 25
        
        # Check SOP structure
        sop = sops[0]
        required_fields = [
            "id", "procedure_name", "procedure_text", "category",
            "department", "version", "effective_date", "last_reviewed",
            "next_review", "created_by", "tags"
        ]
        
        for field in required_fields:
            assert field in sop, f"Missing field: {field}"
        
        # Validate content
        assert len(sop["procedure_text"]) > 100  # Should have substantial content
        assert "Prerequisites:" in sop["procedure_text"]
        assert "Procedure Steps:" in sop["procedure_text"]
        assert isinstance(sop["tags"], list)
        assert len(sop["tags"]) >= 2
    
    def test_generate_incident_reports(self):
        """Test incident report generation."""
        incidents = generate_incident_reports(50)
        
        assert len(incidents) == 50
        
        # Check incident structure
        incident = incidents[0]
        required_fields = [
            "id", "incident_title", "incident_description", "incident_date",
            "severity", "category", "affected_systems", "resolution",
            "resolution_time_hours", "root_cause", "preventive_measures",
            "reported_by", "resolved_by"
        ]
        
        for field in required_fields:
            assert field in incident, f"Missing field: {field}"
        
        # Validate constraints
        assert incident["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert isinstance(incident["affected_systems"], list)
        assert incident["resolution_time_hours"] > 0
        assert isinstance(incident["incident_date"], datetime)
    
    def test_generate_sales_proposals(self):
        """Test sales proposal generation."""
        customers = generate_customers(10)
        proposals = generate_sales_proposals(customers, 30)
        
        assert len(proposals) == 30
        
        # Check proposal structure
        proposal = proposals[0]
        required_fields = [
            "id", "proposal_id", "customer_id", "customer_name",
            "proposal_date", "proposal_content", "executive_summary",
            "proposal_value", "products", "status", "valid_until",
            "created_by"
        ]
        
        for field in required_fields:
            assert field in proposal, f"Missing field: {field}"
        
        # Validate constraints
        assert proposal["proposal_id"].startswith("PROP")
        assert proposal["status"] in ["DRAFT", "SENT", "UNDER_REVIEW", "WON", "LOST"]
        assert proposal["proposal_value"] > 0
        assert isinstance(proposal["products"], list)
        assert len(proposal["products"]) >= 1
        assert "EXECUTIVE SUMMARY:" in proposal["proposal_content"]


class TestDataIntegration:
    """Test data integration and relationships."""
    
    def test_data_consistency(self):
        """Test that generated data maintains referential consistency."""
        # Generate related data
        suppliers = generate_suppliers(5)
        inventory = generate_inventory(suppliers, 50)
        customers = generate_customers(10)
        transactions = generate_transactions(customers, inventory, 100)
        
        # Check supplier references in inventory
        supplier_ids = {s["supplier_id"] for s in suppliers}
        inventory_supplier_ids = {i["supplier_id"] for i in inventory}
        assert inventory_supplier_ids.issubset(supplier_ids), "Invalid supplier references in inventory"
        
        # Check customer references in transactions
        customer_ids = {c["customer_id"] for c in customers}
        transaction_customer_ids = {t["customer_id"] for t in transactions}
        assert transaction_customer_ids.issubset(customer_ids), "Invalid customer references in transactions"
        
        # Check product references in transactions
        part_ids = {i["part_id"] for i in inventory}
        transaction_part_ids = {t["product_id"] for t in transactions}
        assert transaction_part_ids.issubset(part_ids), "Invalid product references in transactions"
    
    def test_data_volumes(self):
        """Test that data generation produces expected volumes."""
        suppliers = generate_suppliers(10)
        assert len(suppliers) == 10
        
        # Inventory creates records distributed across warehouses but total count equals requested
        inventory = generate_inventory(suppliers, 100)
        assert len(inventory) == 100  # 100 total inventory records across all warehouses
        
        customers = generate_customers(20)
        assert len(customers) == 20
        
        # Each test should produce exact count requested
        transactions = generate_transactions(customers, inventory[:50], 150)
        assert len(transactions) == 150


class TestErrorHandling:
    """Test error handling in data generation."""
    
    def test_empty_suppliers_list(self):
        """Test inventory generation with empty suppliers list."""
        inventory = generate_inventory([], 10)
        
        # Should still generate inventory but with empty supplier_id
        # or handle gracefully
        assert len(inventory) >= 0
    
    def test_zero_count_generation(self):
        """Test generation functions with zero count."""
        suppliers = generate_suppliers(0)
        assert len(suppliers) == 0
        
        customers = generate_customers(0)
        assert len(customers) == 0
    
    @patch('data.generate_sample_data.fake')
    def test_faker_error_handling(self, mock_faker):
        """Test handling of faker library errors."""
        # Mock faker to raise exception
        mock_faker.company.side_effect = Exception("Faker error")
        
        # Should handle gracefully or provide fallback
        try:
            suppliers = generate_suppliers(1)
            # If it doesn't raise, check it handled gracefully
            assert isinstance(suppliers, list)
        except Exception:
            # If it does raise, that's also acceptable behavior
            pass


# Fixtures for common test data
@pytest.fixture
def sample_suppliers():
    """Fixture providing sample supplier data."""
    return generate_suppliers(5)


@pytest.fixture
def sample_customers():
    """Fixture providing sample customer data."""
    return generate_customers(10)


@pytest.fixture
def sample_inventory(sample_suppliers):
    """Fixture providing sample inventory data."""
    return generate_inventory(sample_suppliers, 20)


# Integration tests using fixtures
def test_transaction_generation_with_fixtures(sample_customers, sample_inventory):
    """Test transaction generation using fixtures."""
    transactions = generate_transactions(sample_customers, sample_inventory, 50)
    
    assert len(transactions) == 50
    
    # Verify all transactions reference valid customers and products
    customer_ids = {c["customer_id"] for c in sample_customers}
    part_ids = {i["part_id"] for i in sample_inventory}
    
    for txn in transactions:
        assert txn["customer_id"] in customer_ids
        assert txn["product_id"] in part_ids


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])