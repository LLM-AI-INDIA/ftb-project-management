#!/usr/bin/env python3
"""
Test Google Drive Connection
Simple script to test if Google Drive API is working correctly
"""

import os
import sys
from server import GoogleDriveMCPServer

def test_google_drive():
    """Test Google Drive connection and basic operations"""
    
    print("🧪 Google Drive Connection Test")
    print("=" * 40)
    
    try:
        # Create server instance
        print("📡 Creating Google Drive server instance...")
        gdrive_server = GoogleDriveMCPServer()
        
        # Initialize service
        print("🔐 Initializing Google Drive service...")
        gdrive_server.initialize_drive_service()
        print("✅ Google Drive service initialized successfully!")
        
        # Test listing files
        print("\n📁 Testing file listing...")
        files = gdrive_server.list_files(max_results=5)
        print(f"✅ Found {len(files)} files")
        
        if files:
            print("\n📋 Sample files:")
            for i, file in enumerate(files[:3], 1):
                print(f"  {i}. {file['name']} ({file['mimeType']})")
        else:
            print("ℹ️  No supported files found in your Google Drive")
            print("   Supported types: CSV, Excel, PDF, PowerPoint, Word, Text, XML, JSON")
        
        # Test search
        print(f"\n🔍 Testing search functionality...")
        search_files = gdrive_server.list_files(query="test", max_results=3)
        print(f"✅ Search returned {len(search_files)} files matching 'test'")
        
        print("\n🎉 All tests passed! Google Drive integration is working correctly.")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n💡 Solution: Run 'python reset_auth.py' first")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Run 'python reset_auth.py' to reset authentication")
        print("2. Check that credentials.json is in the same directory")
        print("3. Verify Google Drive API is enabled in Google Cloud Console")
        print("4. Make sure OAuth consent screen is configured")
        return False

if __name__ == "__main__":
    success = test_google_drive()
    if success:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
