import os
import sys
import unittest
import io

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.api.app import app

class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_invalid_file_extension(self):
        """Test that non-video files are rejected with 400."""
        data = {
            'video': (io.BytesIO(b"dummy data"), 'test.txt'),
            'task_name': 'Gait Analysis'
        }
        response = self.app.post('/api/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Unsupported file type', response.get_json()['error'])

    def test_missing_video_file(self):
        """Test that missing video file returns 400."""
        data = {'task_name': 'Gait Analysis'}
        response = self.app.post('/api/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No video file provided', response.get_json()['error'])

    def test_empty_filename(self):
        """Test that empty filename returns 400."""
        data = {'video': (io.BytesIO(b""), ''), 'task_name': 'Gait Analysis'}
        response = self.app.post('/api/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No selected file', response.get_json()['error'])

if __name__ == '__main__':
    unittest.main()
