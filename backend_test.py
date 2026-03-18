#!/usr/bin/env python3
"""
Backend API Testing for Bone Fracture Detection System
Tests all API endpoints with comprehensive error handling
"""

import requests
import sys
import os
import json
from datetime import datetime
from pathlib import Path

class BoneFractureAPITester:
    def __init__(self, base_url="https://xray-classifier.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []

    def log_result(self, test_name, success, details=""):
        """Log test result with details"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {test_name} - PASSED")
            if details:
                print(f"   Details: {details}")
        else:
            self.failures.append({"test": test_name, "details": details})
            print(f"❌ {test_name} - FAILED")
            if details:
                print(f"   Error: {details}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, timeout=30):
        """Run a single API test with comprehensive error handling"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'} if not files else {}

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=timeout)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if success:
                try:
                    response_data = response.json()
                    details += f", Response: {json.dumps(response_data, indent=2)[:200]}..."
                except:
                    details += f", Text: {response.text[:100]}..."
            else:
                details += f", Expected: {expected_status}"
                try:
                    error_data = response.json()
                    details += f", Error: {json.dumps(error_data)}"
                except:
                    details += f", Text: {response.text[:200]}"

            self.log_result(name, success, details)
            return success, response.json() if success else {}

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout}s"
            self.log_result(name, False, error_msg)
            return False, {}
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error - server may be down"
            self.log_result(name, False, error_msg)
            return False, {}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.log_result(name, False, error_msg)
            return False, {}

    def test_health_check(self):
        """Test API health check"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_dataset_info(self):
        """Test dataset info endpoint"""
        success, response = self.run_test(
            "Dataset Info",
            "GET", 
            "dataset-info",
            200
        )
        
        if success and response:
            # Validate dataset structure
            expected_keys = ['train', 'val']
            for key in expected_keys:
                if key not in response:
                    self.log_result(f"Dataset Info - {key} key", False, f"Missing {key} in response")
                else:
                    self.log_result(f"Dataset Info - {key} key", True, f"Found {key} with data: {response[key]}")
        
        return success

    def test_model_metrics(self):
        """Test model metrics endpoint (may fail if no trained model)"""
        success, response = self.run_test(
            "Model Metrics",
            "GET",
            "model-metrics", 
            200,
            timeout=10
        )
        
        # This is expected to fail if no trained model exists
        if not success:
            print("   Note: This is expected if no model has been trained yet")
        
        return success

    def test_training_status(self):
        """Test training status endpoint"""
        success, response = self.run_test(
            "Training Status",
            "GET",
            "training-status",
            200
        )
        
        if success and response:
            expected_keys = ['is_training', 'status', 'progress']
            for key in expected_keys:
                if key not in response:
                    self.log_result(f"Training Status - {key} key", False, f"Missing {key} in response")
                else:
                    self.log_result(f"Training Status - {key} key", True, f"{key}: {response[key]}")
        
        return success

    def test_predictions_history(self):
        """Test predictions history endpoint"""
        success, response = self.run_test(
            "Predictions History",
            "GET",
            "predictions-history",
            200
        )
        return success

    def create_test_image(self):
        """Create a simple test image for prediction testing"""
        try:
            from PIL import Image
            import io
            
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color='gray')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            return img_bytes
        except ImportError:
            print("   PIL not available, skipping image creation test")
            return None

    def test_prediction_endpoint(self):
        """Test prediction endpoint with test image"""
        # Try to create a test image
        test_image = self.create_test_image()
        
        if test_image is None:
            # Try to use an existing image from dataset
            data_dir = Path("/app/backend/data")
            test_image_paths = []
            
            for split in ['train', 'val', 'test']:
                split_dir = data_dir / split
                if split_dir.exists():
                    for class_dir in ['normal', 'fractured']:
                        class_path = split_dir / class_dir
                        if class_path.exists():
                            for img_file in class_path.glob('*'):
                                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                    test_image_paths.append(img_file)
                                    break
                        if test_image_paths:
                            break
                if test_image_paths:
                    break
            
            if test_image_paths:
                try:
                    with open(test_image_paths[0], 'rb') as f:
                        test_image = f.read()
                        filename = test_image_paths[0].name
                        print(f"   Using dataset image: {filename}")
                except Exception as e:
                    print(f"   Could not read dataset image: {e}")
                    self.log_result("Prediction Test", False, "No test image available")
                    return False
            else:
                self.log_result("Prediction Test", False, "No dataset images found")
                return False
        else:
            filename = "test_image.jpg"
            test_image = test_image.getvalue()

        # Test prediction endpoint
        files = {'file': (filename, test_image, 'image/jpeg')}
        success, response = self.run_test(
            "X-ray Prediction",
            "POST",
            "predict",
            200,
            files=files,
            timeout=30
        )
        
        if success and response:
            # Validate prediction response structure
            expected_keys = ['prediction', 'confidence', 'probabilities', 'class_idx']
            for key in expected_keys:
                if key not in response:
                    self.log_result(f"Prediction Response - {key}", False, f"Missing {key} in response")
                else:
                    self.log_result(f"Prediction Response - {key}", True, f"{key}: {response[key]}")
        
        return success

    def test_training_endpoint(self):
        """Test training endpoint (but don't actually train to save time)"""
        # Test with invalid parameters first
        success, response = self.run_test(
            "Training Start (Invalid Parameters)", 
            "POST",
            "train",
            422,  # Expected validation error
            data={"epochs": -1, "batch_size": 0, "learning_rate": -0.1}
        )
        
        # Note: We won't test actual training as it takes too long
        print("   Note: Actual training test skipped to save time")
        return True

    def run_all_tests(self):
        """Run all backend API tests"""
        print("=" * 60)
        print("🧪 BONE FRACTURE DETECTION API TESTING")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"API URL: {self.api_url}")
        print()

        # Test each endpoint
        self.test_health_check()
        self.test_dataset_info()
        self.test_training_status()
        self.test_predictions_history()
        self.test_model_metrics()
        self.test_prediction_endpoint()
        self.test_training_endpoint()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        if self.failures:
            print("\n❌ FAILURES:")
            for i, failure in enumerate(self.failures, 1):
                print(f"{i}. {failure['test']}: {failure['details']}")
        else:
            print("\n🎉 All tests passed!")

        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = BoneFractureAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())