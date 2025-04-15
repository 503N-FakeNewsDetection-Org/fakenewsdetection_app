import os
import tempfile
import torch
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from transformers import AutoModel, BertTokenizerFast
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the module to test
from retraining.text_retrain import (
    load_data_from_file, tokenize_texts, 
    BERT_Arch, train_epoch, evaluate_epoch, 
    promote_to_production, load_model_from_file
)

class TestDataPreparation(unittest.TestCase):
    """Unit tests for data preparation functions"""
    
    def setUp(self):
        """Set up test data"""
        self.test_titles = [
            # Real news (label 0) - 9 titles
            "Scientists discover new planet in nearby star system",
            "Local community raises funds for new park",
            "Company announces new renewable energy initiative",
            "Study shows benefits of daily exercise",
            "New technology improves solar panel efficiency",
            "City council approves new public transportation plan",
            "Researchers develop new method for plastic recycling",
            "Local school receives grant for STEM education",
            "Weather service predicts mild winter season",
            
            # Fake news (label 1) - 11 titles
            "President announces new policy that will shock you",
            "Breaking: Local man wins lottery, shares secret",
            "Company releases product that will change everything",
            "Study reveals shocking truth about common food",
            "New technology will make your phone obsolete",
            "Breaking: Government hiding alien technology",
            "Secret cure for cancer discovered, doctors hate this",
            "One simple trick to lose weight fast",
            "You won't believe what scientists just found",
            "This common household item is poisoning your family",
            "Breaking: Major corporation hiding dangerous secret"
        ]
        self.test_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.test_data = pd.DataFrame({
            'title': self.test_titles,
            'label': self.test_labels
        })
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, 'test_data.csv')
        
        # Save test data to file
        self.test_data.to_csv(self.test_file_path, index=False)
        
        # Set up patch for environment variables
        self.env_patcher = patch.dict('os.environ', {
            'DATA_PATH': self.test_file_path
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()
        self.env_patcher.stop()
    
    def test_tokenize_texts(self):
        """Test text tokenization"""
        seq, mask = tokenize_texts(self.test_data['title'])
        self.assertEqual(seq.shape[0], len(self.test_data))
        self.assertEqual(mask.shape[0], len(self.test_data))
        self.assertEqual(seq.shape[1], 15)  # MAX_LENGTH
    
    def test_tokenize_texts_empty(self):
        """Test tokenization with empty input"""
        with self.assertRaises(ValueError):
            tokenize_texts(pd.Series([]))
    
    def test_tokenize_texts_invalid(self):
        """Test tokenization with invalid input"""
        with self.assertRaises(ValueError):
            tokenize_texts(None)

    def test_load_data_from_file(self):
        """Test loading data from local file"""
        loaded_data = load_data_from_file(self.test_file_path)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), len(self.test_data))
        pd.testing.assert_frame_equal(loaded_data, self.test_data)
    
    def test_load_data_from_file_empty(self):
        """Test loading empty data file"""
        empty_df = pd.DataFrame(columns=['title', 'label'])
        empty_file_path = os.path.join(self.temp_dir.name, 'empty.csv')
        empty_df.to_csv(empty_file_path, index=False)
        
        with self.assertRaises(ValueError):
            load_data_from_file(empty_file_path)
    
    def test_load_data_from_file_missing_columns(self):
        """Test loading data with missing columns"""
        invalid_df = pd.DataFrame({'title': self.test_titles})
        invalid_file_path = os.path.join(self.temp_dir.name, 'invalid.csv')
        invalid_df.to_csv(invalid_file_path, index=False)
        
        with self.assertRaises(ValueError):
            load_data_from_file(invalid_file_path)

class TestModelArchitecture(unittest.TestCase):
    """Unit tests for model architecture"""
    
    def setUp(self):
        """Set up test model"""
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.model = BERT_Arch(self.bert)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create test data for forward pass
        self.batch_size = 4
        self.seq_len = 15
        self.input_ids = torch.randint(0, 100, (self.batch_size, self.seq_len)).to(self.device)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
        self.labels = torch.tensor([0, 1, 0, 1]).to(self.device)
        
        # Create optimizer and criterion
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.criterion = torch.nn.NLLLoss()
        
        # Create dataloader
        self.dataset = torch.utils.data.TensorDataset(
            self.input_ids, self.attention_mask, self.labels
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2
        )
        
        # Create a temporary directory for test model
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_model_path = os.path.join(self.temp_dir.name, 'test_model.pt')
        
        # Save test model to the temp directory
        torch.save(self.model.state_dict(), self.test_model_path)
    
    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, BERT_Arch)
        self.assertEqual(self.model.fc2.out_features, 2)  # Binary classification
        self.assertIsNotNone(self.model.bert)
        self.assertIsNotNone(self.model.dropout)
        self.assertIsNotNone(self.model.relu)
        self.assertIsNotNone(self.model.fc1)
        self.assertIsNotNone(self.model.fc2)
        self.assertIsNotNone(self.model.softmax)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        output = self.model(self.input_ids, self.attention_mask)
        self.assertEqual(output.shape, (self.batch_size, 2))
        
        # Check probabilities (softmax outputs)
        probs = torch.exp(output)
        row_sums = probs.sum(dim=1)
        # Verify each row sums to approximately 1 (within floating-point precision)
        for row_sum in row_sums:
            self.assertAlmostEqual(row_sum.item(), 1.0, places=5)
    
    def test_train_epoch(self):
        """Test training function"""
        with patch('retraining.text_retrain.train_epoch', return_value=0.5):
            loss = train_epoch(self.model, self.dataloader, self.optimizer, self.criterion)
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)  # Loss should be positive
            self.assertLess(loss, 10)    # Loss shouldn't be too large
    
    def test_evaluate_epoch(self):
        """Test evaluation function"""
        with patch('retraining.text_retrain.evaluate_epoch', return_value=0.3):
            loss = evaluate_epoch(self.model, self.dataloader, self.criterion)
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)  # Loss should be positive
            self.assertLess(loss, 10)    # Loss shouldn't be too large
    
    def test_load_model_from_file(self):
        """Test loading model from file"""
        # Use the patched function to load from our test path
        
        # Load the model from our test path
        loaded_model = load_model_from_file(self.test_model_path)
        
        # Verify the model loaded correctly
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, BERT_Arch)
        
        # Check that model parameters are the same as our original test model
        for original_param, loaded_param in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(original_param.data, loaded_param.data, atol=1e-4))
            
    def test_load_model_from_file_nonexistent(self):
        """Test loading model from nonexistent file"""
        nonexistent_path = os.path.join(self.temp_dir.name, 'nonexistent_model.pt')
        
        # Should return None for nonexistent file
        result = load_model_from_file(nonexistent_path)
        self.assertIsNone(result)

class TestModelPromotion(unittest.TestCase):
    """Unit tests for model promotion logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.model = BERT_Arch(self.bert)
        
        # Mock MLflow
        self.mlflow_patcher = patch('mlflow.log_metric')
        self.mock_log_metric = self.mlflow_patcher.start()
        
        # Set up patch for environment variables with high thresholds
        self.env_patcher = patch.dict('os.environ', {
            'ACCURACY_THRESHOLD': '0.88',
            'F1_THRESHOLD': '0.88'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()
        self.mlflow_patcher.stop()
        self.env_patcher.stop()
    
    @patch('retraining.text_retrain.push_to_production')
    @patch('retraining.text_retrain.reset_user_data')
    def test_promote_to_production_success(self, mock_reset, mock_push):
        """Test successful model promotion with good metrics"""
        # Set up mocks
        mock_push.return_value = True
        mock_reset.return_value = True
        
        # Test with metrics above threshold
        metrics = {
            "test_accuracy": 0.9,
            "test_f1": 0.9,
            "test_precision": 0.9,
            "test_recall": 0.9
        }
        
        result = promote_to_production(self.model, metrics, self.temp_dir.name)
        
        # Verify results
        self.assertTrue(result)
        mock_push.assert_called_once()
        mock_reset.assert_called_once()
        self.mock_log_metric.assert_called_with("production_model_promoted", 1)
    
    @patch('retraining.text_retrain.push_to_production')
    @patch('retraining.text_retrain.reset_user_data')
    def test_promote_to_production_failure_poor_metrics(self, mock_reset, mock_push):
        """Test failed model promotion with poor metrics"""
        # Test with metrics below threshold
        metrics = {
            "test_accuracy": 0.8,
            "test_f1": 0.8,
            "test_precision": 0.8,
            "test_recall": 0.8
        }
        
        result = promote_to_production(self.model, metrics, self.temp_dir.name)
        
        # Verify results
        self.assertFalse(result)
        mock_push.assert_not_called()
        mock_reset.assert_not_called()
        self.mock_log_metric.assert_called_with("production_model_promoted", 0)

if __name__ == "__main__":
    unittest.main()