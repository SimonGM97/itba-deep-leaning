#!/bin/bash
# chmod +x ./scripts/bash/01_tests_running.sh
# ./scripts/bash/01_tests_running.sh

# Run Data Processing unit & integrity tests
python3 -m unittest test/test_data_processing/test_data_extractor.py
python3 -m unittest test/test_data_processing/test_data_cleaner.py
python3 -m unittest test/test_data_processing/test_data_shifter.py
python3 -m unittest test/test_data_processing/test_data_refiner.py
python3 -m unittest test/test_data_processing/test_feature_selector.py # ERROR!
python3 -m unittest test/test_data_processing/test_data_transformer.py

# # Run Trading unit & integrity tests
# python3 -m unittest test/test_trading/test_trading_table.py
# python3 -m unittest test/test_trading/test_ptxbot.py

# # Run Client unit & integrity tests
# python3 -m unittest test/test_binance_client/test_binance_client.py

# # Run Modeling unit & integrity tests
# python3 -m unittest test/test_modeling/test_model.py
# python3 -m unittest test/test_modeling/test_model_registry.py
# python3 -m unittest test/test_modeling/test_model_tuning.py

# # Run Pipeline unit & integriry tests
# python3 -m unittest test/test_pipeline/test_ml_pipeline.py # ERROR!

# # Run Data Warehousing unit & integrity tests
# python3 -m unittest test/test_data_warehousing/test_data_warehouse_manager.py
# python3 -m unittest test/test_data_warehousing/test_trading_analyst.py # ERROR!