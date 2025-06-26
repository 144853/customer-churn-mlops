"""Data validation module using Great Expectations."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import FileDataContext
from great_expectations.exceptions import DataContextError
from loguru import logger

from ..config.settings import settings


class DataValidator:
    """Data validation using Great Expectations."""
    
    def __init__(self, context_root_dir: str = "great_expectations"):
        self.context_root_dir = Path(context_root_dir)
        self.data_context = self._get_or_create_context()
    
    def _get_or_create_context(self) -> FileDataContext:
        """Get or create Great Expectations data context."""
        try:
            if self.context_root_dir.exists():
                context = FileDataContext(context_root_dir=str(self.context_root_dir))
            else:
                context = FileDataContext.create(project_root_dir=".")
            return context
        except DataContextError as e:
            logger.error(f"Error creating data context: {e}")
            raise
    
    def create_expectation_suite(self, suite_name: str) -> None:
        """Create expectation suite for data validation."""
        try:
            # Customer profiles expectations
            if suite_name == "customer_profiles_suite":
                suite = self.data_context.add_expectation_suite(
                    expectation_suite_name=suite_name
                )
                
                # Add expectations
                expectations = [
                    {
                        "expectation_type": "expect_table_row_count_to_be_between",
                        "kwargs": {"min_value": 1000, "max_value": 10000000}
                    },
                    {
                        "expectation_type": "expect_column_to_exist",
                        "kwargs": {"column": "customer_id"}
                    },
                    {
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "kwargs": {"column": "customer_id"}
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_unique",
                        "kwargs": {"column": "customer_id"}
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {"column": "age", "min_value": 18, "max_value": 120}
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {"column": "gender", "value_set": ["M", "F", "Other"]}
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {"column": "tenure_months", "min_value": 0, "max_value": 240}
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {
                            "column": "contract_type", 
                            "value_set": ["Month-to-month", "One year", "Two year"]
                        }
                    },
                    {
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {"column": "monthly_charges", "min_value": 0, "max_value": 1000}
                    }
                ]
                
                for expectation in expectations:
                    suite.add_expectation(**expectation)
                
                self.data_context.save_expectation_suite(suite)
                logger.info(f"Created expectation suite: {suite_name}")
                
        except Exception as e:
            logger.error(f"Error creating expectation suite {suite_name}: {e}")
            raise
    
    def validate_data(self, 
                     df: pd.DataFrame, 
                     suite_name: str, 
                     batch_identifier: str = "default_batch") -> Dict[str, Any]:
        """Validate dataframe against expectation suite."""
        try:
            logger.info(f"Validating data with suite: {suite_name}")
            
            # Create datasource and data connector
            datasource_config = {
                "name": "pandas_datasource",
                "class_name": "Datasource",
                "execution_engine": {
                    "class_name": "PandasExecutionEngine"
                },
                "data_connectors": {
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["batch_id"]
                    }
                }
            }
            
            self.data_context.add_datasource(**datasource_config)
            
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="validation_data",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"batch_id": batch_identifier}
            )
            
            # Create validator
            validator = self.data_context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Run validation
            validation_result = validator.validate()
            
            # Log results
            success_rate = validation_result.success
            failed_expectations = len([r for r in validation_result.results if not r.success])
            total_expectations = len(validation_result.results)
            
            logger.info(f"Validation completed: {success_rate}, "
                       f"Failed: {failed_expectations}/{total_expectations}")
            
            return {
                "success": validation_result.success,
                "results": validation_result.results,
                "statistics": validation_result.statistics,
                "failed_expectations": failed_expectations,
                "total_expectations": total_expectations
            }
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            raise
    
    def validate_pipeline_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Validate all pipeline data."""
        validation_results = {}
        
        # Validate customer profiles
        if "customer_profiles" in data:
            validation_results["customer_profiles"] = self.validate_data(
                data["customer_profiles"], 
                "customer_profiles_suite",
                "customer_profiles_batch"
            )
        
        return validation_results
