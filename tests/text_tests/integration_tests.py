import os
import unittest
import pandas as pd
import torch
import tempfile
import shutil
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the functions we want to test
from retraining.text_retrain import (
    push_to_production,
    archive_user_data,
    reset_user_data,
    promote_to_production,
    BERT_Arch,
    load_data_from_file,
    load_model_from_file,
    archive_user_data
)


class TextRetrainingIntegrationTests(unittest.TestCase):
    """Integration tests for the text retraining system."""
    
    def setUp(self):
        """Set up test environment and data."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create model directory structure
        self.model_dir = os.path.join(self.test_dir, 'text_model')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Store original environment variables to restore later
        self.original_envs = {
            'DATA_PATH': os.environ.get('DATA_PATH'),
            'ARCHIVE_DIR': os.environ.get('ARCHIVE_DIR'),
        }
        
        # Create sample data with 9 real titles and 11 fake titles
        self.sample_data = pd.DataFrame({
            'title': [
                # Real news items (9)
                'This is real news about politics',
                'True report on economics',
                'Accurate description of events',
                'Factual account of history',
                'Legitimate news about technology',
                'Reliable information about science',
                'Authentic coverage of sports',
                'Genuine reporting on education',
                'Honest journalism about healthcare',
                
                # Fake news items (11)
                'Fake story about celebrities',
                'Fabricated tale about science',
                'Made up narrative about sports',
                'Fictional account of weather',
                'Deceptive article about health',
                'False report on global events',
                'Misleading story about finance',
                'Untrue claims about government',
                'Invented scandal about politicians',
                'Fraudulent news about economy',
                'Artificial hype about technology'
            ],
            'label': [
                # 9 real news items (0)
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                
                # 11 fake news items (1)
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        })
        
        # Verify label distribution
        assert sum(self.sample_data['label']) == 11, "Expected 11 fake news items"
        assert len(self.sample_data) - sum(self.sample_data['label']) == 9, "Expected 9 real news items"
        
        # Set up environment variables for testing
        self.original_env = {
            'DATA_PATH': os.environ.get('DATA_PATH'),
            'ARCHIVE_DIR': os.environ.get('ARCHIVE_DIR'),
            'AZURE_STORAGE_CONNECTION_STRING': os.environ.get('AZURE_STORAGE_CONNECTION_STRING'),
            'MODELS_CONTAINER': os.environ.get('MODELS_CONTAINER'),
            'CSV_CONTAINER': os.environ.get('CSV_CONTAINER'),
            'MODEL_BLOB': os.environ.get('MODEL_BLOB'),
            'CSV_BLOB': os.environ.get('CSV_BLOB'),
            'ACCURACY_THRESHOLD': os.environ.get('ACCURACY_THRESHOLD'),
            'F1_THRESHOLD': os.environ.get('F1_THRESHOLD')
        }
        
        # Set test environment variables
        os.environ['ACCURACY_THRESHOLD'] = '0.85'
        os.environ['F1_THRESHOLD'] = '0.85'
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_archive_user_data(self):
        """Test that user data is properly archived by copying."""
        # Create a project-like directory structure in the temp directory
        project_dir = os.path.join(self.test_dir, 'text_IEP')
        datasets_dir = os.path.join(project_dir, 'datasets')
        data_dir = os.path.join(datasets_dir, 'text_data')
        archive_dir = os.path.join(data_dir, 'archives') # Define archive dir path
        os.makedirs(archive_dir, exist_ok=True) # Create archive dir
        
        # Create a test user data file
        test_file_path = os.path.join(data_dir, 'user_data.csv')
        self.sample_data.to_csv(test_file_path, index=False)
        
        # Call the archive function with our test paths
        result = archive_user_data(test_file_path, archive_dir)
        
        # Verify the function was successful
        self.assertTrue(result, "Archive function returned False, indicating an error")
        
        # Check that archive file was created in the specified archive directory
        archive_files = [f for f in os.listdir(archive_dir) if f.startswith('user_data_') and f.endswith('.csv')]
        self.assertEqual(len(archive_files), 1, f"Expected 1 archive file in {archive_dir}, found {len(archive_files)}")
        
        # Verify the content of the archive by reading it directly
        archive_path = os.path.join(archive_dir, archive_files[0])
        # Use pd.read_csv directly on the archive file, not load_data_from_file
        archived_data_direct = pd.read_csv(archive_path)

        # Sort both dataframes to ensure comparison is order-independent
        expected_data_sorted = self.sample_data.sort_values(by='title').reset_index(drop=True)
        archived_data_sorted = archived_data_direct.sort_values(by='title').reset_index(drop=True)

        # Compare the directly read archive data with the original sample data
        pd.testing.assert_frame_equal(archived_data_sorted, expected_data_sorted)
        self.assertEqual(len(archived_data_direct), 20) # Check length explicitly
        self.assertEqual(sum(archived_data_direct['label']), 11) # Check label sum explicitly
    
    @patch('retraining.text_retrain.BlobServiceClient')
    def test_push_to_production(self, mock_blob_service):
        """Test that model can be pushed to production storage."""
        # Set up mocks for Azure storage
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client
        
        # Create a mock model
        bert = MagicMock()
        bert.return_value = {'pooler_output': torch.rand(1, 768)}
        model = BERT_Arch(bert)
        
        # Test pushing to production
        result = push_to_production(model)
        
        # Verify Azure blob interactions
        self.assertTrue(mock_blob_client.stage_block.called)
        self.assertTrue(mock_blob_client.commit_block_list.called)
    
    @patch('retraining.text_retrain.BlobServiceClient')
    @patch('retraining.text_retrain.archive_user_data')
    def test_reset_user_data(self, mock_archive, mock_blob_service):
        """Test that user data can be reset after archiving."""
        # Set up mocks
        mock_archive.return_value = True
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client
        
        # Call function
        result = reset_user_data()
        
        # Check results
        self.assertTrue(result)
        mock_archive.assert_called_once()
        mock_blob_client.upload_blob.assert_called_once()
    
    @patch('mlflow.log_metric')
    @patch('retraining.text_retrain.push_to_production')
    @patch('retraining.text_retrain.reset_user_data')
    def test_promote_to_production_success(self, mock_reset, mock_push, mock_log_metric):
        """Test successful model promotion with MLflow logging."""
        # Set up mocks
        mock_push.return_value = True
        mock_reset.return_value = True
        
        # Create a temporary model file
        model_dir = os.path.join(self.test_dir, 'text_model')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pt')
        
        # Create a simple model and save it
        bert = MagicMock()
        bert.return_value = {'pooler_output': torch.rand(1, 768)}
        model = BERT_Arch(bert)
        torch.save(model.state_dict(), model_path)
        
        # Try loading the model with our custom path 
        loaded_model = load_model_from_file(model_path)
        self.assertIsNotNone(loaded_model, "Failed to load model from test path")
        
        # Good metrics that should trigger promotion
        good_metrics = {
            "test_accuracy": 0.90,
            "test_f1": 0.90,
            "test_precision": 0.89,
            "test_recall": 0.91
        }
        
        # Test with good metrics using the loaded model
        result = promote_to_production(loaded_model, good_metrics, self.test_dir)
        
        # Check results
        self.assertTrue(result)
        mock_push.assert_called_once()
        mock_reset.assert_called_once()
        mock_log_metric.assert_called_with("production_model_promoted", 1)
    
    @patch('mlflow.log_metric')
    @patch('retraining.text_retrain.push_to_production')
    @patch('retraining.text_retrain.reset_user_data')
    def test_promote_to_production_failure(self, mock_reset, mock_push, mock_log_metric):
        """Test failed model promotion with MLflow logging."""
        # Create a temporary model file
        model_dir = os.path.join(self.test_dir, 'text_model')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pt')
        
        # Create and save a test model
        bert = MagicMock()
        bert.return_value = {'pooler_output': torch.rand(1, 768)}
        model = BERT_Arch(bert)
        torch.save(model.state_dict(), model_path)
        
        # Load the model with our custom path
        loaded_model = load_model_from_file(model_path)
        self.assertIsNotNone(loaded_model, "Failed to load model from test path")
        
        # Poor metrics that should not trigger promotion
        poor_metrics = {
            "test_accuracy": 0.80,
            "test_f1": 0.80,
            "test_precision": 0.80,
            "test_recall": 0.80
        }
        
        # Test with poor metrics using the loaded model
        result = promote_to_production(loaded_model, poor_metrics, self.test_dir)
        
        # Check results
        self.assertFalse(result)
        mock_push.assert_not_called()
        mock_reset.assert_not_called()
        mock_log_metric.assert_called_with("production_model_promoted", 0)


if __name__ == '__main__':
    unittest.main()
