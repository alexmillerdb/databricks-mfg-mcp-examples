#!/usr/bin/env python3
"""
Sample Data Generation Script for Manufacturing MCP Examples
Phase 2.2: Generate Sample Data

This script generates realistic synthetic data for all Delta tables:
- 10,000+ inventory items across 5 warehouses
- 1,000+ active shipments with various statuses
- 500+ customers with purchase history
- 100,000+ IoT sensor readings
- 1,000+ support tickets with resolutions

Can be run from Databricks notebooks or locally via Databricks Connect.
"""

import os
import sys
import random
import uuid
from datetime import datetime, timedelta, date
from typing import List, Dict, Any

# Try to import faker, install if not available
try:
    from faker import Faker
except ImportError:
    print("Installing faker library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faker"])
    from faker import Faker

fake = Faker()
Faker.seed(42)  # For reproducibility
random.seed(42)


def setup_databricks_environment():
    """Universal setup for local IDE or notebook execution"""
    
    # Detect execution environment
    is_notebook = 'DATABRICKS_RUNTIME_VERSION' in os.environ
    
    if is_notebook:
        print("🟢 Databricks Notebook Environment")
        return setup_notebook_env()
    else:
        print("🔵 Local IDE Environment")
        return setup_local_env()


def setup_notebook_env():
    """Setup for Databricks notebook"""
    from databricks.sdk import WorkspaceClient
    
    # In notebook, spark is available globally
    global spark  # Declare global to avoid undefined variable warning
    return {
        'environment': 'notebook',
        'spark': spark,  # Available globally in notebooks
        'workspace_client': WorkspaceClient(),
        'catalog': os.getenv('UC_DEFAULT_CATALOG', 'mfg_mcp_demo'),
        'schema': 'supply_chain'  # Default schema
    }


def setup_local_env():
    """Setup for local IDE"""
    from dotenv import load_dotenv
    from databricks.connect import DatabricksSession
    from databricks.sdk import WorkspaceClient
    import mlflow
    
    # Load environment variables
    load_dotenv()
    
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
    catalog = os.getenv("UC_DEFAULT_CATALOG", "mfg_mcp_demo")
    schema = os.getenv("UC_DEFAULT_SCHEMA", "supply_chain")
    
    try:
        # Initialize Databricks Connect
        spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()
        
        # Set catalog context only (no schema since we use fully qualified table names)
        spark.sql(f"USE CATALOG {catalog}")
        
        # Configure MLflow (optional for this script but good to have)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
        mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
        
        return {
            'environment': 'local',
            'spark': spark,
            'workspace_client': WorkspaceClient(profile=profile),
            'catalog': catalog,
            'schema': schema
        }
    except Exception as e:
        print(f"⚠️  Error setting up local environment: {e}")
        print("   Running in dry-run mode (data will be generated but not loaded)")
        return {
            'environment': 'local',
            'spark': None,  # Dry-run mode
            'workspace_client': None,
            'catalog': catalog,
            'schema': schema
        }


def cleanup_environment(config):
    """Clean up resources (local only)"""
    if config.get('environment') == 'local' and config.get('spark'):
        try:
            config['spark'].stop()
        except:
            pass  # Ignore errors during cleanup


def generate_suppliers(num_suppliers: int = 50) -> List[Dict[str, Any]]:
    """Generate supplier data."""
    suppliers = []
    
    supplier_types = [
        "Manufacturing", "Electronics", "Components", "Raw Materials",
        "Packaging", "Logistics", "Assembly", "Tooling"
    ]
    
    for i in range(num_suppliers):
        supplier_id = f"SUP{str(i+1).zfill(4)}"
        
        supplier = {
            "supplier_id": supplier_id,
            "supplier_name": fake.company() + " " + random.choice(supplier_types),
            "contact_name": fake.name(),
            "contact_email": fake.company_email(),
            "contact_phone": fake.phone_number(),
            "address": fake.street_address(),
            "city": fake.city(),
            "country": fake.country(),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "lead_time_days": int(random.randint(3, 30)),  # Ensure Python int
            "payment_terms": random.choice(["Net 30", "Net 60", "Net 90", "COD", "2/10 Net 30"]),
            "status": random.choice(["ACTIVE"] * 9 + ["INACTIVE"]),  # 90% active
            "created_date": fake.date_between(start_date="-5y", end_date="-1y"),
            "last_order_date": fake.date_between(start_date="-6m", end_date="today")
        }
        suppliers.append(supplier)
    
    return suppliers


def generate_inventory(suppliers: List[Dict], num_items: int = 10000) -> List[Dict[str, Any]]:
    """Generate inventory data across warehouses."""
    inventory = []
    
    # Handle empty suppliers list
    if not suppliers:
        print("Warning: No suppliers provided, using default supplier")
        suppliers = [{"supplier_id": "DEFAULT_SUP"}]
    
    warehouses = [
        ("WH001", "Chicago, IL"),
        ("WH002", "Los Angeles, CA"),
        ("WH003", "Houston, TX"),
        ("WH004", "Atlanta, GA"),
        ("WH005", "Seattle, WA")
    ]
    
    part_categories = [
        "Electronics", "Mechanical", "Hydraulic", "Pneumatic",
        "Electrical", "Fasteners", "Raw Material", "Consumables"
    ]
    
    part_prefixes = ["BOLT", "NUT", "GEAR", "MOTOR", "SENSOR", "VALVE", 
                     "PUMP", "FILTER", "BEARING", "SEAL", "CABLE", "SWITCH"]
    
    # Generate unique parts (distribute evenly across warehouses)
    parts_per_warehouse = num_items // len(warehouses)
    
    # Generate parts
    parts = []
    for i in range(parts_per_warehouse):
        part_id = f"PART{str(i+1).zfill(5)}"
        part_name = f"{random.choice(part_prefixes)}-{fake.lexify('????').upper()}-{random.randint(100,999)}"
        parts.append((part_id, part_name))
    
    # Create inventory records for each part in each warehouse
    for warehouse_id, warehouse_location in warehouses:
        for part_id, part_name in parts:
            current_qty = random.randint(0, 1000)
            reorder_level = random.randint(50, 200)
            
            record = {
                "inventory_id": str(uuid.uuid4()),
                "part_id": part_id,
                "part_name": part_name,
                "warehouse_id": warehouse_id,
                "warehouse_location": warehouse_location,
                "current_quantity": int(current_qty),  # Ensure Python int
                "reorder_level": int(reorder_level),  # Ensure Python int
                "reorder_quantity": int(reorder_level * 3),  # Ensure Python int
                "unit_cost": round(random.uniform(0.50, 500.00), 2),
                "last_updated": datetime.now() - timedelta(hours=random.randint(0, 72)),
                "category": random.choice(part_categories),
                "supplier_id": random.choice(suppliers)["supplier_id"]
            }
            inventory.append(record)
    
    return inventory


def generate_shipments(inventory: List[Dict], suppliers: List[Dict], 
                      num_shipments: int = 1000) -> List[Dict[str, Any]]:
    """Generate shipment data."""
    shipments = []
    
    statuses = ["PENDING", "IN_TRANSIT", "DELIVERED", "DELAYED", "CANCELLED"]
    carriers = ["FedEx", "UPS", "DHL", "USPS", "Local Courier", "Company Fleet"]
    
    for i in range(num_shipments):
        item = random.choice(inventory)
        supplier = random.choice(suppliers)
        
        shipment_date = fake.date_between(start_date="-3m", end_date="today")
        expected_days = random.randint(1, 14)
        expected_delivery = shipment_date + timedelta(days=expected_days)
        
        status = random.choice(statuses)
        actual_delivery = None
        if status == "DELIVERED":
            actual_delivery = expected_delivery + timedelta(days=random.randint(-2, 3))
        
        shipment = {
            "shipment_id": f"SHP{str(i+1).zfill(6)}",
            "order_id": f"ORD{str(i+1).zfill(6)}",
            "part_id": item["part_id"],
            "part_name": item["part_name"],
            "quantity": int(random.randint(10, 500)),  # Ensure Python int
            "supplier_id": supplier["supplier_id"],
            "supplier_name": supplier["supplier_name"],
            "origin_location": supplier["city"],
            "destination_location": item["warehouse_location"],
            "shipment_date": shipment_date,
            "expected_delivery": expected_delivery,
            "actual_delivery": actual_delivery,
            "status": status,
            "carrier": random.choice(carriers),
            "tracking_number": fake.lexify("????????").upper() + str(random.randint(100000, 999999)),
            "cost": round(random.uniform(50.00, 2000.00), 2)
        }
        shipments.append(shipment)
    
    return shipments


def generate_customers(num_customers: int = 500) -> List[Dict[str, Any]]:
    """Generate customer data."""
    customers = []
    
    industries = ["Automotive", "Aerospace", "Electronics", "Construction", 
                  "Energy", "Healthcare", "Retail", "Manufacturing"]
    segments = ["Enterprise", "SMB", "Startup"]
    
    for i in range(num_customers):
        created_date = fake.date_between(start_date="-5y", end_date="-1m")
        total_orders = random.randint(1, 100)
        avg_order_value = random.uniform(1000, 50000)
        total_spent = round(total_orders * avg_order_value, 2)
        
        # Determine status based on last purchase
        days_since_purchase = random.randint(1, 365)
        if days_since_purchase > 180:
            status = random.choice(["INACTIVE", "CHURNED"])
        else:
            status = "ACTIVE"
        
        last_purchase = date.today() - timedelta(days=days_since_purchase)
        
        customer = {
            "customer_id": f"CUST{str(i+1).zfill(5)}",
            "customer_name": fake.company(),
            "contact_name": fake.name(),
            "email": fake.company_email(),
            "phone": fake.phone_number(),
            "industry": random.choice(industries),
            "segment": random.choice(segments),
            "address": fake.street_address(),
            "city": fake.city(),
            "country": fake.country(),
            "created_date": created_date,
            "status": status,
            "total_orders": int(total_orders),  # Ensure Python int
            "total_spent": total_spent,
            "last_purchase_date": last_purchase,
            "credit_limit": round(random.uniform(10000, 500000), 2),
            "payment_terms": random.choice(["Net 30", "Net 60", "Net 90"])
        }
        customers.append(customer)
    
    return customers


def generate_transactions(customers: List[Dict], inventory: List[Dict], 
                         num_transactions: int = 5000) -> List[Dict[str, Any]]:
    """Generate sales transaction data."""
    transactions = []
    
    sales_reps = [(f"SR{str(i+1).zfill(3)}", fake.name()) for i in range(20)]
    regions = ["North", "South", "East", "West", "Central"]
    payment_methods = ["Credit Card", "Wire Transfer", "Check", "ACH", "Purchase Order"]
    
    for i in range(num_transactions):
        customer = random.choice(customers)
        item = random.choice(inventory)
        sales_rep = random.choice(sales_reps)
        
        quantity = random.randint(1, 100)
        unit_price = item["unit_cost"] * random.uniform(1.2, 2.0)  # Markup
        total_amount = round(quantity * unit_price, 2)
        discount_rate = random.choice([0, 0.05, 0.1, 0.15, 0.2])
        discount_amount = round(total_amount * discount_rate, 2)
        tax_rate = 0.08
        tax_amount = round((total_amount - discount_amount) * tax_rate, 2)
        
        transaction = {
            "transaction_id": f"TXN{str(i+1).zfill(7)}",
            "order_id": f"ORD{str(i+1).zfill(6)}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["customer_name"],
            "transaction_date": fake.date_time_between(start_date="-6m", end_date="now"),
            "product_id": item["part_id"],
            "product_name": item["part_name"],
            "quantity": int(quantity),  # Ensure Python int
            "unit_price": round(unit_price, 2),
            "total_amount": total_amount,
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "payment_method": random.choice(payment_methods),
            "sales_rep_id": sales_rep[0],
            "sales_rep_name": sales_rep[1],
            "region": random.choice(regions),
            "status": random.choice(["COMPLETED"] * 9 + ["PENDING"])  # 90% completed
        }
        transactions.append(transaction)
    
    return transactions


def generate_iot_telemetry(num_readings: int = 100000) -> List[Dict[str, Any]]:
    """Generate IoT sensor telemetry data."""
    telemetry = []
    
    # Define devices
    devices = []
    device_types = ["Temperature Sensor", "Pressure Sensor", "Flow Meter", 
                    "Vibration Monitor", "Power Meter"]
    locations = ["Production Line A", "Production Line B", "Warehouse 1", 
                 "Warehouse 2", "Quality Control", "Packaging"]
    
    for i in range(50):  # 50 devices
        devices.append({
            "device_id": f"DEV{str(i+1).zfill(4)}",
            "device_type": random.choice(device_types),
            "location": random.choice(locations)
        })
    
    # Generate readings
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_readings):
        device = random.choice(devices)
        timestamp = start_date + timedelta(seconds=i * 30)  # Reading every 30 seconds
        
        # Generate realistic values based on device type
        if "Temperature" in device["device_type"]:
            temperature = random.gauss(25, 5)  # Normal around 25°C
            humidity = random.gauss(50, 10)
            pressure = None
            vibration = None
            power = random.uniform(0.1, 1.0)
        elif "Pressure" in device["device_type"]:
            temperature = random.gauss(30, 3)
            humidity = None
            pressure = random.gauss(100, 20)  # PSI
            vibration = None
            power = random.uniform(0.5, 2.0)
        else:
            temperature = random.gauss(25, 5)
            humidity = random.gauss(50, 10)
            pressure = random.gauss(100, 20)
            vibration = random.uniform(0, 10)
            power = random.uniform(1.0, 10.0)
        
        # Occasionally generate anomalies
        operational_status = "NORMAL"
        error_code = None
        maintenance_required = False
        
        if random.random() < 0.05:  # 5% chance of anomaly
            operational_status = random.choice(["WARNING", "ERROR"])
            error_code = f"E{random.randint(100, 999)}"
            maintenance_required = random.choice([True, False])
        
        reading = {
            "reading_id": str(uuid.uuid4()),
            "device_id": device["device_id"],
            "device_type": device["device_type"],
            "location_id": device["location"],
            "timestamp": timestamp,
            "temperature": round(temperature, 2) if temperature else None,
            "humidity": round(humidity, 2) if humidity else None,
            "pressure": round(pressure, 2) if pressure else None,
            "vibration": round(vibration, 2) if vibration else None,
            "power_consumption": round(power, 2),
            "operational_status": operational_status,
            "error_code": error_code,
            "maintenance_required": maintenance_required,
            "reading_date": timestamp.date()
        }
        telemetry.append(reading)
    
    return telemetry


def generate_support_tickets(customers: List[Dict], num_tickets: int = 1000) -> List[Dict[str, Any]]:
    """Generate support ticket data."""
    tickets = []
    
    categories = ["Technical", "Billing", "Shipping", "Product Quality", 
                  "Feature Request", "General Inquiry"]
    priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    statuses = ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED", "ESCALATED"]
    
    support_agents = [fake.name() for _ in range(10)]
    
    ticket_templates = [
        ("Equipment malfunction", "The {equipment} is showing error code {code}. Production is affected."),
        ("Delayed shipment", "Order {order} has not arrived. Expected delivery was {date}."),
        ("Quality issue", "Batch {batch} has quality defects. {percent}% failure rate observed."),
        ("Missing parts", "Order {order} is missing {count} units of {part}."),
        ("Technical support", "Need assistance with {system} configuration."),
        ("Billing discrepancy", "Invoice {invoice} shows incorrect amount. Expected {expected} but charged {actual}."),
    ]
    
    for i in range(num_tickets):
        customer = random.choice(customers)
        template = random.choice(ticket_templates)
        
        created_date = fake.date_time_between(start_date="-6m", end_date="now")
        
        # Determine resolution based on age
        age_days = (datetime.now() - created_date).days
        if age_days > 7:
            status = random.choice(["RESOLVED", "CLOSED"])
            resolved_date = created_date + timedelta(hours=random.randint(1, 168))
            resolution_hours = (resolved_date - created_date).total_seconds() / 3600
            satisfaction = random.choice([3, 4, 4, 5, 5])  # Mostly positive
        else:
            status = random.choice(["OPEN", "IN_PROGRESS", "ESCALATED"])
            resolved_date = None
            resolution_hours = None
            satisfaction = None
        
        # Generate ticket content
        subject = template[0]
        description = template[1].format(
            equipment=fake.word(),
            code=f"E{random.randint(100, 999)}",
            order=f"ORD{random.randint(100000, 999999)}",
            date=fake.date(),
            batch=f"B{random.randint(1000, 9999)}",
            percent=random.randint(5, 25),
            count=random.randint(1, 50),
            part=fake.word(),
            system=fake.word(),
            invoice=f"INV{random.randint(10000, 99999)}",
            expected=random.randint(1000, 10000),
            actual=random.randint(1100, 11000)
        )
        
        # Generate tags as a list of strings (ensure consistency for type inference)
        available_tags = ["urgent", "warranty", "recurring", "vip", "escalated", "refund"]
        num_tags = random.randint(0, 3)
        ticket_tags = random.sample(available_tags, k=num_tags) if num_tags > 0 else []
        
        ticket = {
            "ticket_id": f"TKT{str(i+1).zfill(6)}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["customer_name"],
            "created_date": created_date,
            "category": random.choice(categories),
            "priority": random.choices(priorities, weights=[30, 40, 20, 10])[0],
            "subject": subject,
            "description": description,
            "status": status,
            "assigned_to": random.choice(support_agents),
            "resolved_date": resolved_date,
            "resolution_time_hours": round(resolution_hours, 2) if resolution_hours else None,
            "satisfaction_score": int(satisfaction) if satisfaction else None,  # Ensure int type
            "tags": ticket_tags  # Consistent array type
        }
        tickets.append(ticket)
    
    return tickets


def generate_sop_documents(num_docs: int = 100) -> List[Dict[str, Any]]:
    """Generate Standard Operating Procedures documents for Vector Search."""
    sops = []
    
    categories = ["Safety", "Quality Control", "Maintenance", "Operations", 
                  "Emergency Response", "Training", "Compliance"]
    departments = ["Production", "Warehouse", "Quality", "Maintenance", 
                   "Safety", "HR", "Engineering"]
    
    sop_templates = [
        ("Machine Startup Procedure", "Step-by-step process for safely starting production equipment"),
        ("Quality Inspection Protocol", "Detailed inspection criteria and acceptance standards"),
        ("Emergency Shutdown Process", "Critical steps for emergency equipment shutdown"),
        ("Preventive Maintenance Schedule", "Regular maintenance tasks and intervals"),
        ("Inventory Counting Procedure", "Process for cycle counting and annual inventory"),
        ("Safety Equipment Check", "Daily safety equipment inspection checklist"),
        ("Waste Disposal Protocol", "Proper handling and disposal of industrial waste"),
    ]
    
    for i in range(num_docs):
        template = random.choice(sop_templates)
        
        # Generate detailed procedure text
        procedure_text = f"{template[1]}\n\n"
        procedure_text += "Prerequisites:\n"
        for j in range(random.randint(2, 4)):
            procedure_text += f"- {fake.sentence()}\n"
        
        procedure_text += "\nProcedure Steps:\n"
        for j in range(random.randint(5, 10)):
            procedure_text += f"{j+1}. {fake.sentence()}\n"
        
        procedure_text += "\nSafety Considerations:\n"
        for j in range(random.randint(2, 3)):
            procedure_text += f"- {fake.sentence()}\n"
        
        created_date = fake.date_between(start_date="-2y", end_date="-6m")
        
        # Generate tags consistently for type inference
        available_tags = ["critical", "safety", "quality", "mandatory", "iso9001", "osha"]
        num_tags = random.randint(2, 4)
        sop_tags = random.sample(available_tags, k=num_tags)
        
        sop = {
            "id": str(uuid.uuid4()),
            "procedure_name": f"{template[0]} - Rev {random.randint(1, 5)}",
            "procedure_text": procedure_text,
            "category": random.choice(categories),
            "department": random.choice(departments),
            "version": f"{random.randint(1, 3)}.{random.randint(0, 9)}",
            "effective_date": created_date,
            "last_reviewed": created_date + timedelta(days=random.randint(30, 180)),
            "next_review": date.today() + timedelta(days=random.randint(30, 365)),
            "created_by": fake.name(),
            "tags": sop_tags  # Consistent array type
        }
        sops.append(sop)
    
    return sops


def generate_incident_reports(num_incidents: int = 200) -> List[Dict[str, Any]]:
    """Generate incident reports for Vector Search."""
    incidents = []
    
    severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    categories = ["Equipment Failure", "Safety", "Quality", "Environmental", 
                  "Security", "Process Deviation", "Supply Chain"]
    
    incident_scenarios = [
        ("Production line stoppage", "Conveyor belt motor failure caused unexpected downtime"),
        ("Quality defect detected", "Batch inspection revealed dimensional tolerances exceeded"),
        ("Safety near-miss", "Forklift operator nearly collided with pedestrian in warehouse"),
        ("Chemical spill", "Minor hydraulic fluid leak detected in pressing area"),
        ("Power outage", "Unexpected power loss affected production for 2 hours"),
        ("Supplier delivery failure", "Critical components not delivered on scheduled date"),
    ]
    
    for i in range(num_incidents):
        scenario = random.choice(incident_scenarios)
        incident_date = fake.date_time_between(start_date="-1y", end_date="-1d")
        
        # Generate detailed description
        description = f"{scenario[1]}\n\n"
        description += f"Initial Detection: {fake.sentence()}\n"
        description += f"Immediate Impact: {fake.sentence()}\n"
        description += f"Actions Taken: {fake.sentence()}\n"
        
        # Resolution details
        resolution = f"Immediate Response:\n"
        for j in range(random.randint(2, 4)):
            resolution += f"- {fake.sentence()}\n"
        resolution += f"\nPermanent Fix: {fake.sentence()}"
        
        resolution_hours = random.uniform(0.5, 72)
        
        # Generate affected systems consistently for type inference  
        available_systems = ["Production Line A", "Production Line B", "Warehouse", "Quality Lab", "Shipping"]
        num_systems = random.randint(1, 3)
        affected_systems = random.sample(available_systems, k=num_systems)
        
        incident = {
            "id": str(uuid.uuid4()),
            "incident_title": scenario[0],
            "incident_description": description,
            "incident_date": incident_date,
            "severity": random.choices(severities, weights=[40, 30, 20, 10])[0],
            "category": random.choice(categories),
            "affected_systems": affected_systems,  # Consistent array type
            "resolution": resolution,
            "resolution_time_hours": round(resolution_hours, 2),
            "root_cause": fake.sentence(),
            "preventive_measures": fake.paragraph(),
            "reported_by": fake.name(),
            "resolved_by": fake.name()
        }
        incidents.append(incident)
    
    return incidents


def generate_sales_proposals(customers: List[Dict], num_proposals: int = 300) -> List[Dict[str, Any]]:
    """Generate sales proposals for Vector Search."""
    proposals = []
    
    statuses = ["DRAFT", "SENT", "UNDER_REVIEW", "WON", "LOST"]
    products = ["Industrial Automation Package", "Quality Control System", 
                "Predictive Maintenance Solution", "IoT Sensor Suite",
                "Supply Chain Optimization Platform", "Custom Manufacturing Line"]
    
    for i in range(num_proposals):
        customer = random.choice(customers)
        proposal_date = fake.date_between(start_date="-1y", end_date="today")
        
        # Generate proposal content
        executive_summary = f"Proposal for {random.choice(products)} implementation at {customer['customer_name']}. "
        executive_summary += fake.paragraph()
        
        proposal_content = f"EXECUTIVE SUMMARY:\n{executive_summary}\n\n"
        proposal_content += f"BUSINESS CASE:\n{fake.paragraph()}\n\n"
        proposal_content += f"PROPOSED SOLUTION:\n{fake.paragraph()}\n\n"
        proposal_content += f"IMPLEMENTATION TIMELINE:\n"
        for phase in range(1, random.randint(3, 6)):
            proposal_content += f"Phase {phase}: {fake.sentence()}\n"
        proposal_content += f"\nINVESTMENT SUMMARY:\n{fake.paragraph()}"
        
        proposal_value = round(random.uniform(50000, 2000000), 2)
        
        # Determine status based on age
        age_days = (date.today() - proposal_date).days
        if age_days > 60:
            status = random.choice(["WON", "LOST", "LOST"])
        elif age_days > 30:
            status = random.choice(["UNDER_REVIEW", "SENT"])
        else:
            status = random.choice(["DRAFT", "SENT"])
        
        # Generate products consistently for type inference
        num_products = random.randint(1, 3)
        selected_products = random.sample(products, k=num_products)
        
        proposal = {
            "id": str(uuid.uuid4()),
            "proposal_id": f"PROP{str(i+1).zfill(5)}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["customer_name"],
            "proposal_date": proposal_date,
            "proposal_content": proposal_content,
            "executive_summary": executive_summary,
            "proposal_value": proposal_value,
            "products": selected_products,  # Consistent array type
            "status": status,
            "valid_until": proposal_date + timedelta(days=90),
            "created_by": fake.name(),
            "win_probability": round(random.uniform(0.1, 0.9), 2) if status in ["SENT", "UNDER_REVIEW"] else None,
            "competitor_info": fake.company() if random.random() > 0.5 else None,
            "notes": fake.sentence() if random.random() > 0.7 else None
        }
        proposals.append(proposal)
    
    return proposals


def write_to_delta_table(spark, data: List[Dict], table_name: str, catalog: str = "mfg_mcp_demo"):
    """Write data to Delta table."""
    if not spark or not data:
        return False
    
    try:
        from pyspark.sql import Row
        from pyspark.sql.functions import col
        from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, TimestampType
        
        # Special handling for tables with complex types
        if "tickets" in table_name:
            # Define explicit schema for support tickets to avoid type inference issues
            schema = None
            try:
                # Get the existing table schema first
                existing_df = spark.table(f"{catalog}.{table_name}")
                schema = existing_df.schema
                df = spark.createDataFrame([Row(**record) for record in data], schema=schema)
                print(f"   ℹ️  Using existing table schema")
            except Exception:
                # Fall back to creating with inferred schema but ensure arrays are properly typed
                df = spark.createDataFrame([Row(**record) for record in data])
                print(f"   ℹ️  Using inferred schema")
        else:
            # Convert to DataFrame
            df = spark.createDataFrame([Row(**record) for record in data])
        
        # Get the existing table schema if it exists and cast if needed
        try:
            existing_df = spark.table(f"{catalog}.{table_name}")
            target_schema = existing_df.schema
            
            # Cast DataFrame columns to match target schema
            for field in target_schema.fields:
                if field.name in df.columns:
                    df = df.withColumn(field.name, col(field.name).cast(field.dataType))
            
            print(f"   ℹ️  Casting data types to match existing table schema")
            
        except Exception:
            # Table doesn't exist or can't read schema, proceed with original DataFrame
            pass
        
        # Write to Delta table 
        if "tickets" in table_name:
            # For support tickets, try insertInto to use existing table schema
            try:
                df.write.mode("overwrite").insertInto(f"{catalog}.{table_name}")
                print(f"   ✅ Wrote {len(data)} records to {catalog}.{table_name} (insertInto)")
                return True
            except Exception as insert_error:
                print(f"   ⚠️  insertInto failed, trying saveAsTable: {insert_error}")
                # Fall back to saveAsTable
                df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{table_name}")
                print(f"   ✅ Wrote {len(data)} records to {catalog}.{table_name} (saveAsTable)")
                return True
        else:
            df.write.mode("overwrite").saveAsTable(f"{catalog}.{table_name}")
            print(f"   ✅ Wrote {len(data)} records to {catalog}.{table_name}")
            return True
        
    except Exception as e:
        print(f"   ❌ Error writing to {catalog}.{table_name}: {e}")
        return False


def main():
    """Main execution function to generate and load sample data."""
    print("=" * 60)
    print("Sample Data Generation for Manufacturing MCP Examples")
    print("Phase 2.2: Generating Sample Data")
    print("=" * 60)
    
    # Setup environment (universal for notebook or local)
    config = setup_databricks_environment()
    spark = config.get('spark')
    catalog_name = config.get('catalog', 'mfg_mcp_demo')
    
    if not spark:
        print("\n⚠️  No Spark session available.")
        print("   Data will be generated but not loaded to Delta tables.")
    
    print(f"\n📦 Target catalog: {catalog_name}")
    print("-" * 40)
    
    # Generate data
    print("\n📊 Generating sample data...")
    
    print("   Generating suppliers...")
    suppliers = generate_suppliers(50)
    print(f"   ✅ Generated {len(suppliers)} suppliers")
    
    print("   Generating inventory...")
    inventory = generate_inventory(suppliers, 10000)
    print(f"   ✅ Generated {len(inventory)} inventory items")
    
    print("   Generating shipments...")
    shipments = generate_shipments(inventory, suppliers, 1000)
    print(f"   ✅ Generated {len(shipments)} shipments")
    
    print("   Generating customers...")
    customers = generate_customers(500)
    print(f"   ✅ Generated {len(customers)} customers")
    
    print("   Generating transactions...")
    transactions = generate_transactions(customers, inventory, 5000)
    print(f"   ✅ Generated {len(transactions)} transactions")
    
    print("   Generating IoT telemetry...")
    telemetry = generate_iot_telemetry(100000)
    print(f"   ✅ Generated {len(telemetry)} telemetry readings")
    
    print("   Generating support tickets...")
    tickets = generate_support_tickets(customers, 1000)
    print(f"   ✅ Generated {len(tickets)} support tickets")
    
    print("   Generating SOP documents...")
    sops = generate_sop_documents(100)
    print(f"   ✅ Generated {len(sops)} SOP documents")
    
    print("   Generating incident reports...")
    incidents = generate_incident_reports(200)
    print(f"   ✅ Generated {len(incidents)} incident reports")
    
    print("   Generating sales proposals...")
    proposals = generate_sales_proposals(customers, 300)
    print(f"   ✅ Generated {len(proposals)} sales proposals")
    
    # Load data to Delta tables if Spark is available
    if spark:
        print("\n💾 Loading data to Delta tables...")
        print("-" * 40)
        
        # Use catalog
        try:
            spark.sql(f"USE CATALOG {catalog_name}")
        except Exception as e:
            print(f"   ❌ Could not use catalog {catalog_name}: {e}")
            return 1
        
        # Load data
        write_to_delta_table(spark, suppliers, "supply_chain.suppliers", catalog_name)
        write_to_delta_table(spark, inventory, "supply_chain.inventory", catalog_name)
        write_to_delta_table(spark, shipments, "supply_chain.shipments", catalog_name)
        write_to_delta_table(spark, sops, "supply_chain.standard_operating_procedures", catalog_name)
        write_to_delta_table(spark, incidents, "supply_chain.incident_reports", catalog_name)
        
        write_to_delta_table(spark, customers, "sales.customers", catalog_name)
        write_to_delta_table(spark, transactions, "sales.transactions", catalog_name)
        write_to_delta_table(spark, proposals, "sales.sales_proposals", catalog_name)
        
        write_to_delta_table(spark, telemetry, "iot.telemetry", catalog_name)
        
        write_to_delta_table(spark, tickets, "support.tickets", catalog_name)
        # Create enhanced support tickets for vector search
        support_tickets = []
        for t in tickets:
            # Ensure tags is always a list for consistent type inference
            ticket_tags = t["tags"] if isinstance(t["tags"], list) else []
            
            enhanced_ticket = {
                "id": str(uuid.uuid4()),
                "ticket_id": t["ticket_id"],
                "customer_id": t["customer_id"],
                "customer_name": t["customer_name"],
                "ticket_content": f"{t['subject']}\n\n{t['description']}",
                "problem_description": t["description"],
                "resolution": f"Status: {t['status']}. Assigned to: {t['assigned_to']}",
                "category": t["category"],
                "product_affected": "Manufacturing System",
                "created_date": t["created_date"],
                "resolved_date": t["resolved_date"],
                "priority": t["priority"],
                "tags": ticket_tags
            }
            support_tickets.append(enhanced_ticket)
        write_to_delta_table(spark, support_tickets, "support.support_tickets", catalog_name)
        
        print("\n✅ Sample data generation and loading completed!")
    else:
        print("\n✅ Sample data generated (not loaded - no Spark session)")
    
    print(f"\n📊 Summary:")
    print(f"   - Suppliers: {len(suppliers)}")
    print(f"   - Inventory Items: {len(inventory)}")
    print(f"   - Shipments: {len(shipments)}")
    print(f"   - Customers: {len(customers)}")
    print(f"   - Transactions: {len(transactions)}")
    print(f"   - IoT Readings: {len(telemetry)}")
    print(f"   - Support Tickets: {len(tickets)}")
    print(f"   - SOP Documents: {len(sops)}")
    print(f"   - Incident Reports: {len(incidents)}")
    print(f"   - Sales Proposals: {len(proposals)}")
    
    print("\n" + "=" * 60)
    print("Next step: Run create_vector_indexes.py to set up Vector Search")
    print("=" * 60)
    
    # Clean up resources if running locally
    cleanup_environment(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())